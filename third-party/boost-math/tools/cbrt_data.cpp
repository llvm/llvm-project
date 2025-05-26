//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <fstream>
#include "mp_t.hpp"
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;
using namespace std;

struct cube_data_generator
{
   mp_t operator()(mp_t z)
   {
      mp_t result = z*z*z;
      // if result is out of range of a float, 
      // don't include in test data as it messes up our results:
      if(result > (std::numeric_limits<float>::max)())
         throw std::domain_error("");
      if(result < (std::numeric_limits<float>::min)())
         throw std::domain_error("");
      return result;
   }
};

int main(int argc, char* argv[])
{
   parameter_info<mp_t> arg1;
   test_data<mp_t> data;

   std::cout << "Welcome.\n"
      "This program will generate spot tests for the cbrt function:\n\n";

   bool cont;
   std::string line;

   do{
      if(0 == get_user_parameter_info(arg1, "z"))
         return 1;
      data.insert(cube_data_generator(), arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
   }while(cont);

   std::cout << "Enter name of test data file [default=cbrt_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "cbrt_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, "cbrt_data");
   
   return 0;
}




