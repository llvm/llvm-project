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

struct hypergeometric_1f2_gen
{
   mp_t operator()(mp_t a1, mp_t b1, mp_t b2, mp_t z)
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
         term /= b1++;
         term /= b2++;
         term /= k;
         term *= z;
      } while (true);
      //
      // check precision:
      //
      if (abs_result * boost::math::tools::epsilon<mp_t>() / fabs(result) > 1e-40)
         throw std::domain_error("Unable to calculate result");

      std::cout << a1 << " " << b1 << " " << b2 << " " << z << " " << result << std::endl;
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
   bool cont = true;

   while (cont)
   {
      float range;
      std::cout << "Enter the range to calculate over for a1, b1 and b2 (single value, range will be -x to x): ";
      std::cin >> range;

      float z_range;
      std::cout << "Enter the range to calculate over for z (single value, range will be -x to x): ";
      std::cin >> z_range;

      int num_spots;
      std::cout << "Enter how many test points to calculate: ";
      std::cin >> num_spots;

      std::vector<mp_t> v;
      random_ns::mt19937 rnd;
      random_ns::uniform_real_distribution<float> ur_a(-range, range);
      random_ns::uniform_real_distribution<float> ur_z(-z_range, z_range);

      do
      {
         mp_t a1 = ur_a(rnd);
         mp_t b1 = ur_a(rnd);
         mp_t b2 = ur_a(rnd);
         mp_t z = ur_z(rnd);

         arg1 = make_single_param(a1);
         arg2 = make_single_param(b1);
         arg3 = make_single_param(b2);
         arg4 = make_single_param(z);
         data.insert(hypergeometric_1f2_gen(), arg1, arg2, arg3, arg4);
      } while (num_spots--);

      std::cout << "Any more data?";
      std::cin >> cont;

   }



   std::cout << "Enter name of test data file [default=hypergeometric_1f2.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "hypergeometric_1f2.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());
   
   return 0;
}


