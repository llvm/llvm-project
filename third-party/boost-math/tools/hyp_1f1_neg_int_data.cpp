//  (C) Copyright John Maddock 2023.
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

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

struct hypergeometric_1f1_gen
{
   mp_t operator()(mp_t arg1, mp_t arg2, mp_t z)
   {
      boost::multiprecision::mpfr_float a1(arg1), a2(arg2), a3(z), r;
      r = boost::math::hypergeometric_pFq_precision({ a1 }, { a2 }, a3, 50);
      return mp_t(r);
   }
};


int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for 1F1 with negative integer arguments:\n";

   std::string line;
   bool cont;

   random_ns::mt19937 rnd;
   random_ns::uniform_real_distribution<float> ur_a(0, 20);


   for (int i = -4; i > -100000; i *= 7)
   {
      arg1 = make_single_param(mp_t(i));
      arg2 = make_single_param(mp_t(-100004));
      arg3 = make_single_param(mp_t(ur_a(rnd)));

      data.insert(hypergeometric_1f1_gen(), arg1, arg2, arg3);
      arg3.z1 = -arg3.z1;
      data.insert(hypergeometric_1f1_gen(), arg1, arg2, arg3);
   }

   for (int i = -4; i > -100000; i *= 7)
   {
      arg1 = make_single_param(mp_t(i));
      arg2 = make_single_param(mp_t(mp_t(i) - 3));
      arg3 = make_single_param(mp_t(ur_a(rnd)));

      data.insert(hypergeometric_1f1_gen(), arg1, arg2, arg3);
      arg3.z1 = -arg3.z1;
      data.insert(hypergeometric_1f1_gen(), arg1, arg2, arg3);
   }

   std::cout << "Enter name of test data file [default=hypergeometric_1f1_neg_int.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if (line == "")
      line = "hypergeometric_1f1_neg_int.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}


