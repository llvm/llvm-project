//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/hypergeometric_2F0.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

struct hypergeometric_2f0_gen
{
   mp_t operator()(mp_t a1, mp_t a2, mp_t z)
   {
      std::cout << a1 << " " << a2 << " " << z << std::endl;
      mp_t result = boost::math::detail::hypergeometric_2F0_generic_series(a1, a2, z, boost::math::policies::policy<>());
      std::cout << a1 << " " << a2 << " " << z << " " << result << std::endl;
      return result;
   }
};

struct hypergeometric_2f0_gen_spec1
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t> operator()(mp_t a1, mp_t z)
   {
      mp_t result = boost::math::detail::hypergeometric_2F0_generic_series(a1, a1 + 0.5, z, boost::math::policies::policy<>());
      std::cout << a1 << " " << a1 + 0.5 << " " << z << " " << result << std::endl;
      return boost::math::make_tuple(a1, a1 + 0.5, z, result);
   }
};

int main(int, char* [])
{
   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for 2F0:\n";

   std::string line;

#if 1
   arg1 = make_periodic_param(mp_t(-20), mp_t(-1), 19);
   arg2 = make_random_param(mp_t(-5), mp_t(5), 8);
   arg1.type |= dummy_param;
   arg2.type |= dummy_param;
   data.insert(hypergeometric_2f0_gen_spec1(), arg1, arg2);


#else

   bool cont;
   do {
      get_user_parameter_info(arg1, "a1");
      get_user_parameter_info(arg2, "a2");
      get_user_parameter_info(arg3, "z");
      data.insert(hypergeometric_2f0_gen(), arg1, arg2, arg3);
      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   } while (cont);

#endif
   std::cout << "Enter name of test data file [default=hypergeometric_2f0.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "hypergeometric_2f0.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());
   
   return 0;
}


