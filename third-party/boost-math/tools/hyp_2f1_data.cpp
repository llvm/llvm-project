//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_MAX_SERIES_ITERATION_POLICY 10000000

#include "mp_t.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

struct hypergeometric_2f1_gen
{
   mp_t operator()(mp_t a1, mp_t a2, mp_t b, mp_t z)
   {
      mp_t result = 0;
      mp_t abs_result = 0;
      mp_t term = 1;
      mp_t k = 0;

      do
      {
         result += term;
         abs_result += fabs(term);
         if (fabs(result) * boost::math::tools::epsilon<mp_t>() > fabs(term))
            break;
         ++k;
         term *= a1++;
         term *= a2++;
         term /= b++;
         term /= k;
         term *= z;
      } while (true);
      //
      // check precision:
      //
      if (abs_result * boost::math::tools::epsilon<mp_t>() / fabs(result) > 1e-40)
         throw std::domain_error("Unable to calculate result");

      std::cout << a1 << " " << a2 << " " << b << " " << z << " " << result << std::endl;
      return result;
   }
};

int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3, arg4;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for 2F0:\n";

   std::string line;

   std::vector<mp_t> v;
   random_ns::mt19937 rnd;
   random_ns::uniform_real_distribution<float> ur_a(-100, 100);
   random_ns::uniform_real_distribution<float> ur_z(-1, 1);

   int num_spots;
   std::cout << "Enter how many test points to calculate: ";
   std::cin >> num_spots;

   do
   {
      mp_t a1 = ur_a(rnd);
      mp_t a2 = ur_a(rnd);
      mp_t b = ur_a(rnd);
      mp_t z = ur_z(rnd);

      arg1 = make_single_param(a1);
      arg2 = make_single_param(a2);
      arg3 = make_single_param(b);
      arg4 = make_single_param(z);
      data.insert(hypergeometric_2f1_gen(), arg1, arg2, arg3, arg4);
   } while (num_spots--);


   std::cout << "Enter name of test data file [default=hypergeometric_2f1.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "hypergeometric_2f1.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());
   
   return 0;
}

