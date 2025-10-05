//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;


template<class T>
boost::math::tuple<T, T, T, T> legendre_p_data(T n, T x)
{
   n = floor(n);
   T r1 = legendre_p(boost::math::tools::real_cast<int>(n), x);
   T r2 = legendre_q(boost::math::tools::real_cast<int>(n), x);
   return boost::math::make_tuple(n, x, r1, r2);
}
   
template<class T>
boost::math::tuple<T, T, T, T> assoc_legendre_p_data(T n, T x)
{
   static boost::mt19937 r;
   int l = real_cast<int>(floor(n));
   boost::uniform_int<> ui((std::max)(-l, -40), (std::min)(l, 40));
   int m = ui(r);
   T r1 = legendre_p(l, m, x);
   return boost::math::make_tuple(n, m, x, r1);
}

int main(int argc, char*argv [])
{
   using namespace boost::math::tools;

   parameter_info<mp_t> arg1, arg2;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   if(argc < 1)
      return 1;

   if(strcmp(argv[1], "--legendre2") == 0)
   {
      do{
         if(0 == get_user_parameter_info(arg1, "l"))
            return 1;
         if(0 == get_user_parameter_info(arg2, "x"))
            return 1;
         arg1.type |= dummy_param;
         arg2.type |= dummy_param;

         data.insert(&legendre_p_data<mp_t>, arg1, arg2);

         std::cout << "Any more data [y/n]?";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         cont = (line == "y");
      }while(cont);
   }
   else if(strcmp(argv[1], "--legendre3") == 0)
   {
      do{
         if(0 == get_user_parameter_info(arg1, "l"))
            return 1;
         if(0 == get_user_parameter_info(arg2, "x"))
            return 1;
         arg1.type |= dummy_param;
         arg2.type |= dummy_param;

         data.insert(&assoc_legendre_p_data<mp_t>, arg1, arg2);

         std::cout << "Any more data [y/n]?";
         std::getline(std::cin, line);
         boost::algorithm::trim(line);
         cont = (line == "y");
      }while(cont);
   }


   std::cout << "Enter name of test data file [default=legendre_p.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "legendre_p.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);
   line.erase(line.find('.'));
   write_code(ofs, data, line.c_str());

   return 0;
}

