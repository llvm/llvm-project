//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/constants/constants.hpp>
#include <map>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace boost::math;

//
// This program calculates the coefficients of the polynomials
// used for the regularized incomplete gamma functions gamma_p
// and gamma_q when parameter a is large, and sigma is small
// (where sigma = fabs(1 - x/a) ).
//
// See "The Asymptotic Expansion of the Incomplete Gamma Functions"
// N. M. Temme.
// Siam J. Math Anal. Vol 10 No 4, July 1979, p757.
// Coefficient calculation is described from Eq 3.8 (p762) onwards.
//

//
// Alpha:
//
mp_t alpha(unsigned k)
{
   static map<unsigned, mp_t> data;
   if(data.empty())
   {
      data[1] = 1;
   }

   map<unsigned, mp_t>::const_iterator pos = data.find(k);
   if(pos != data.end())
      return (*pos).second;
   //
   // OK try and calculate the value:
   //
   mp_t result = alpha(k-1);
   for(unsigned j = 2; j <= k-1; ++j)
   {
      result -= j * alpha(j) * alpha(k-j+1);
   }
   result /= (k+1);
   data[k] = result;
   return result;
}

mp_t gamma(unsigned k)
{
   static map<unsigned, mp_t> data;

   map<unsigned, mp_t>::const_iterator pos = data.find(k);
   if(pos != data.end())
      return (*pos).second;

   mp_t result = (k&1) ? -1 : 1;

   for(unsigned i = 1; i <= (2 * k + 1); i += 2)
      result *= i;
   result *= alpha(2 * k + 1);
   data[k] = result;
   return result;
}

mp_t Coeff(unsigned n, unsigned k)
{
   map<unsigned, map<unsigned, mp_t> > data;
   if(data.empty())
      data[0][0] = mp_t(-1) / 3;

   map<unsigned, map<unsigned, mp_t> >::const_iterator p1 = data.find(n);
   if(p1 != data.end())
   {
      map<unsigned, mp_t>::const_iterator p2 = p1->second.find(k);
      if(p2 != p1->second.end())
      {
         return p2->second;
      }
   }

   //
   // If we don't have the value, calculate it:
   //
   if(k == 0)
   {
      // special case:
      mp_t result = (n+2) * alpha(n+2);
      data[n][k] = result;
      return result;
   }
   // general case:
   mp_t result = gamma(k) * Coeff(n, 0) + (n+2) * Coeff(n+2, k-1);
   data[n][k] = result;
   return result;
}

void calculate_terms(double sigma, double a, unsigned bits)
{
   cout << endl << endl;
   cout << "Sigma:        " << sigma << endl;
   cout << "A:            " << a << endl;
   double lambda = 1 - sigma;
   cout << "Lambda:       " << lambda << endl;
   double y = a * (-sigma - log1p(-sigma));
   cout << "Y:            " << y << endl;
   double z = -sqrt(2 * (-sigma - log1p(-sigma)));
   cout << "Z:            " << z << endl;
   double dom = erfc(sqrt(y)) / 2;
   cout << "Erfc term:    " << dom << endl;
   double lead = exp(-y) / sqrt(2 * constants::pi<double>() * a);
   cout << "Remainder factor: " << lead << endl;
   double eps = ldexp(1.0, 1 - static_cast<int>(bits));
   double target = dom * eps / lead;
   cout << "Target smallest term: " << target << endl;

   unsigned max_n = 0;

   for(unsigned n = 0; n < 10000; ++n)
   {
      double term = tools::real_cast<double>(Coeff(n, 0) * pow(z, (double)n));
      if(fabs(term) < target)
      {
         max_n = n-1;
         break;
      }
   }
   cout << "Max n required:  " << max_n << endl;

   unsigned max_k = 0;
   for(unsigned k = 1; k < 10000; ++k)
   {
      double term = tools::real_cast<double>(Coeff(0, k) * pow(a, -((double)k)));
      if(fabs(term) < target)
      {
         max_k = k-1;
         break;
      }
   }
   cout << "Max k required:  " << max_k << endl << endl;

   bool code = false;
   cout << "Print code [0|1]? ";
   cin >> code;

   std::cout << std::scientific << std::setprecision(40);

   if(code)
   {
      cout << "   T workspace[" << max_k+1 << "];\n\n";
      for(unsigned k = 0; k <= max_k; ++k)
      {
         cout <<
            "   static const T C" << k << "[] = {\n";
         for(unsigned n = 0; n < 10000; ++n)
         {
            double term = tools::real_cast<double>(Coeff(n, k) * pow(a, -((double)k)) * pow(z, (double)n));
            if(fabs(term) < target)
            {
               break;
            }
            cout << "      " << Coeff(n, k) << "L,\n";
         }
         cout << 
            "   };\n"
            "   workspace[" << k << "] = tools::evaluate_polynomial(C" << k << ", z);\n\n";
      }
      cout << "   T result = tools::evaluate_polynomial(workspace, 1/a);\n\n";
   }
}


int main()
{
   bool cont;
   do{
      cont  = false;
      double sigma;
      cout << "Enter max value for sigma (sigma = |1 - x/a|): ";
      cin >> sigma;
      double a;
      cout << "Enter min value for a: ";
      cin >> a;
      unsigned precision;
      cout << "Enter number of bits precision required: ";
      cin >> precision;

      calculate_terms(sigma, a, precision);

      cout << "Try again[0|1]: ";
      cin >> cont;

   }while(cont);


   return 0;
}

