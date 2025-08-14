//  Copyright John Maddock 2008.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/constants/constants.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;
using namespace std;

struct asinh_data_generator
{
   mp_t operator()(mp_t z)
   {
      std::cout << z << " ";
      mp_t result = log(z + sqrt(z * z + 1));
      std::cout << result << std::endl;
      return result;
   }
};

struct acosh_data_generator
{
   mp_t operator()(mp_t z)
   {
      std::cout << z << " ";
      mp_t result = log(z + sqrt(z * z - 1));
      std::cout << result << std::endl;
      return result;
   }
};

struct atanh_data_generator
{
   mp_t operator()(mp_t z)
   {
      std::cout << z << " ";
      mp_t result = log((z + 1) / (1 - z)) / 2;
      std::cout << result << std::endl;
      return result;
   }
};

int main(int argc, char*argv [])
{
   parameter_info<mp_t> arg1;
   test_data<mp_t> data;
   std::ofstream ofs;

   bool cont;
   std::string line;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the inverse hyperbolic sin function:\n";

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      data.insert(asinh_data_generator(), arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=asinh_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "asinh_data.ipp";
   ofs.open(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "asinh_data");
   data.clear();

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the inverse hyperbolic cos function:\n";

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      data.insert(acosh_data_generator(), arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=acosh_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "acosh_data.ipp";
   ofs.close();
   ofs.open(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "acosh_data");
   data.clear();

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the inverse hyperbolic tan function:\n";

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      data.insert(atanh_data_generator(), arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=atanh_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "atanh_data.ipp";
   ofs.close();
   ofs.open(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "atanh_data");
   
   return 0;
}

