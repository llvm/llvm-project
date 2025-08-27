//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/tools/test_data.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;


template<class T>
boost::math::tuple<T, T, T> laguerre2_data(T n, T x)
{
   n = floor(n);
   T r1 = laguerre(boost::math::tools::real_cast<unsigned>(n), x);
   return boost::math::make_tuple(n, x, r1);
}
   
template<class T>
boost::math::tuple<T, T, T, T> laguerre3_data(T n, T m, T x)
{
   n = floor(n);
   m = floor(m);
   T r1 = laguerre(real_cast<unsigned>(n), real_cast<unsigned>(m), x);
   return boost::math::make_tuple(n, m, x, r1);
}

int main(int argc, char*argv [])
{
   using namespace boost::math::tools;

   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   if(argc < 2)
      return 1;

   if(strcmp(argv[1], "--laguerre2") == 0)
   {
      do{
         if(0 == get_user_parameter_info(arg1, "n"))
            return 1;
         if(0 == get_user_parameter_info(arg2, "x"))
            return 1;
         arg1.type |= dummy_param;
         arg2.type |= dummy_param;

         data.insert(&laguerre2_data<mp_t>, arg1, arg2);

         std::cout << "Any more data [y/n]?";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         cont = (line == "y");
      }while(cont);
   }
   else if(strcmp(argv[1], "--laguerre3") == 0)
   {
      do{
         if(0 == get_user_parameter_info(arg1, "n"))
            return 1;
         if(0 == get_user_parameter_info(arg2, "m"))
            return 1;
         if(0 == get_user_parameter_info(arg3, "x"))
            return 1;
         arg1.type |= dummy_param;
         arg2.type |= dummy_param;
         arg3.type |= dummy_param;

         data.insert(&laguerre3_data<mp_t>, arg1, arg2, arg3);

         std::cout << "Any more data [y/n]?";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         cont = (line == "y");
      }while(cont);
   }


   std::cout << "Enter name of test data file [default=laguerre.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "laguerre.ipp";
   std::ofstream ofs(line.c_str());
   line.erase(line.find('.'));
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}

