from scipy.odr import odrpack as odr
from scipy.odr import models

def poly_lsq(x,y,n,verbose=False,itmax=200):
    ''' Performs a polynomial least squares fit to the data,
    with errors! Uses scipy odrpack, but for least squares.
    
    IN:
       x,y (arrays) - data to fit
       n (int)      - polinomial order
       verbose      - can be 0,1,2 for different levels of output
                      (False or True are the same as 0 or 1)
       itmax (int)  - optional maximum number of iterations
       
    OUT:
       coeff -  polynomial coefficients, lowest order first
       err   - standard error (1-sigma) on the coefficients

    --Tiago, 20071114
    '''

    # http://www.scipy.org/doc/api_docs/SciPy.odr.odrpack.html
    # see models.py and use ready made models!!!!
    
    func   = models.polynomial(n)
    mydata = odr.Data(x, y)
    myodr  = odr.ODR(mydata, func,maxit=itmax)

    # Set type of fit to least-squares:
    myodr.set_job(fit_type=2)
    if verbose == 2: myodr.set_iprint(final=2)
          
    fit = myodr.run()

    # Display results:
    if verbose: fit.pprint()

    if fit.stopreason[0] == 'Iteration limit reached':
        print '(WWW) poly_lsq: Iteration limit reached, result not reliable!'

    # Results and errors
    coeff = fit.beta[::-1]
    err   = fit.sd_beta[::-1]

    return coeff,err
