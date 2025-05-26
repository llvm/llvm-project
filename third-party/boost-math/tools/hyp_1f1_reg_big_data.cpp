//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_MAX_SERIES_ITERATION_POLICY 10000000
#define BOOST_MATH_USE_MPFR
#include "mp_t.hpp"
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

#include <boost/multiprecision/mpfr.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;
using namespace boost::multiprecision;


struct hypergeometric_1f1_gen
{
   mp_t operator()(mp_t a1, mp_t a2, mp_t z)
   {
      mp_t result;
      try {
         result = (mp_t)boost::math::hypergeometric_pFq_precision({ mpfr_float(a1) }, { mpfr_float(mpfr_float(a2)) }, mpfr_float(z), 50, 25.0) / boost::multiprecision::tgamma(a2);
         std::cout << a1 << " " << a2 << " " << z << " " << result << std::endl;
      }
      catch (...)
      {
         throw std::domain_error("");
      }
      if (fabs(result) > (std::numeric_limits<double>::max)())
      {
         std::cout << "Rejecting over large value\n";
         throw std::domain_error("");
      }
      return result;
   }
};


int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for 1F1 (Yeh!!):\n";

   std::string line;
   //bool cont;

   std::vector<mp_t> v;
   random_ns::mt19937 rnd;
   random_ns::uniform_real_distribution<float> ur_a(0, 1);

   mp_t p = ur_a(rnd);
   p *= 1e6;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e5;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e4;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e3;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e2;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e-5;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e-12;
   v.push_back(p);
   v.push_back(-p);
   p = ur_a(rnd);
   p *= 1e-30;
   v.push_back(p);
   v.push_back(-p);

   for (unsigned i = 0; i < v.size(); ++i)
   {
      for (unsigned j = 0; j < v.size(); ++j)
      {
         for (unsigned k = 0; k < v.size(); ++k)
         {
            std::cout << i << " " << j << " " << k << std::endl;
            std::cout << v[i] << " " << (v[j] * 3) / 2 << " " << (v[j] * 5) / 4 << std::endl;
            arg1 = make_single_param(v[i]);
            arg2 = make_single_param(mp_t((v[j] * 3) / 2));
            arg3 = make_single_param(mp_t((v[k] * 5) / 4));
            data.insert(hypergeometric_1f1_gen(), arg1, arg2, arg3);
         }
      }
   }


   std::cout << "Enter name of test data file [default=hypergeometric_1f1.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "hypergeometric_1f1.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}


