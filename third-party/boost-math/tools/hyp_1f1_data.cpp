//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_MAX_SERIES_ITERATION_POLICY 10000000

#include "mp_t.hpp"
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
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
   mp_t operator()(mp_t a1, mp_t a2, mp_t z)
   {
      long long scaling = 0;
      std::cout << a1 << " " << a2 << " " << z << std::endl;
      mp_t result = boost::math::detail::hypergeometric_1F1_generic_series(a1, a2, z, boost::math::policies::policy<>(), scaling, "");
      std::cout << a1 << " " << a2 << " " << z << " " << result << std::endl;
      return ldexp(result, scaling);
   }
};

struct hypergeometric_1f1_gen_2
{
   mp_t operator()(mp_t a1, mp_t a2, mp_t z)
   {
      long long scaling = 0;
      mp_t result = boost::math::detail::hypergeometric_1F1_generic_series(a1, a2, z, boost::math::policies::policy<>(), scaling, "");
      result = ldexp(result, scaling);
      std::cout << a1 << " " << a2 << " " << z << " " << result << std::endl;
      if (fabs(result) > (std::numeric_limits<double>::max)())
      {
         std::cout << "Discarding result as too large\n";
         throw std::domain_error("");
      }
      if (static_cast<double>(result) == 1)
      {
         std::cout << "Discarding result as unity\n";
         throw std::domain_error("");  // uninteresting result.
      }
      return result;
   }
};


int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for 2F0:\n";

   std::string line;
   bool cont;

#if 1
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
            arg1 = make_single_param(v[i]);
            arg2 = make_single_param(v[j] * 3 / 2);
            arg3 = make_single_param(v[k] * 5 / 4);
            data.insert(hypergeometric_1f1_gen_2(), arg1, arg2, arg3);
         }
      }
   }


#else

   do {
      get_user_parameter_info(arg1, "a1");
      get_user_parameter_info(arg2, "a2");
      get_user_parameter_info(arg3, "z");
      data.insert(hypergeometric_1f1_gen(), arg1, arg2, arg3);
      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   } while (cont);

#endif
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


