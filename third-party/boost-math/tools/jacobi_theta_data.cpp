/*
 * Copyright Evan Miller, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include "mp_t.hpp"
#include <boost/math/tools/test_data.hpp>
#include <boost/test/included/prg_exec_monitor.hpp>
#include <boost/math/special_functions/jacobi_theta.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

struct jacobi_theta_data_generator
{
   boost::math::tuple<mp_t, mp_t, mp_t, mp_t> operator()(mp_t z, mp_t tau)
   {
      return boost::math::make_tuple(
              jacobi_theta1tau(z, tau),
              jacobi_theta2tau(z, tau),
              jacobi_theta3tau(z, tau),
              jacobi_theta4tau(z, tau));
   }
};

int cpp_main(int argc, char*argv [])
{
   parameter_info<mp_t> arg1, arg2;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   if(argc < 1)
      return 1;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the Jacobi Theta functions.\n"
      ;

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      if(0 == get_user_parameter_info(arg2, "tau"))
         return 1;

      data.insert(jacobi_theta_data_generator(), arg1, arg2);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   } while(cont);

   std::cout << "Generating " << data.size() << " test points.";

   std::cout << "Enter name of test data file [default=jacobi_theta.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "jacobi_theta.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "jacobi_theta_data");

   return 0;
}
