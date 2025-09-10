//  (C) Copyright John Maddock 2009.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//#define BOOST_MATH_INSTRUMENT

#include "mp_t.hpp"
#include <boost/math/distributions/hypergeometric.hpp>
#include <boost/math/special_functions/trunc.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/random/uniform_int.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;

std::mt19937 rnd;

struct hypergeometric_generator
{
   boost::math::tuple<
      mp_t, 
      mp_t, 
      mp_t, 
      mp_t, 
      mp_t,
      mp_t,
      mp_t> operator()(mp_t rN, mp_t rr, mp_t rn)
   {
      using namespace std;
      using namespace boost;
      using namespace boost::math;

      if((rr > rN) || (rr < rn))
         throw std::domain_error("");

      try{
         int N = itrunc(rN);
         int r = itrunc(rr);
         int n = itrunc(rn);
         boost::uniform_int<> ui((std::max)(0, n + r - N), (std::min)(n, r));
         int k = ui(rnd);

         hypergeometric_distribution<mp_t> d(r, n, N);

         mp_t p = pdf(d, k);
         if((p == 1) || (p == 0))
         {
            // trivial case, don't clutter up our table with it:
            throw std::domain_error("");
         }
         mp_t c = cdf(d, k);
         mp_t cc = cdf(complement(d, k));

         std::cout << "N = " << N << " r = " << r << " n = " << n << " PDF = " << p << " CDF = " << c << " CCDF = " << cc << std::endl;

         return boost::math::make_tuple(r, n, N, k, p, c, cc);
      }
      catch(const std::exception& e)
      {
         std::cout << e.what() << std::endl;
         throw std::domain_error("");
      }
   }
};

int main(int argc, char*argv [])
{
   std::string line;
   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests hypergeoemtric distribution:\n";

   arg1 = make_power_param(mp_t(0), 1, 21);
   arg2 = make_power_param(mp_t(0), 1, 21);
   arg3 = make_power_param(mp_t(0), 1, 21);

   arg1.type |= dummy_param;
   arg2.type |= dummy_param;
   arg3.type |= dummy_param;

   data.insert(hypergeometric_generator(), arg1, arg2, arg3);

   line = "hypergeometric_dist_data2.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "hypergeometric_dist_data2");
   
   return 0;
}

