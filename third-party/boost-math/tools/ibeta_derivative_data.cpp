//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "mp_t.hpp"
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <map>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;

#include <table_type.hpp>

#define T double
#define SC_(x) static_cast<double>(x)
#include <ibeta_int_data.ipp>

int main(int, char* [])
{
   std::cout << "Enter name of test data file [default=ibeta_derivative_data.ipp]";
   std::string line;
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "ibeta_derivative_data.ipp";
   std::ofstream ofs(line.c_str());
   ofs << std::scientific << std::setprecision(40);

   ofs <<
      "//  (C) Copyright John Maddock 2006.\n"
      "//  Use, modification and distribution are subject to the\n"
      "//  Boost Software License, Version 1.0. (See accompanying file\n"
      "//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)\n"
      "\n\n"
      "static const std::array<std::array<typename table_type<T>::type, 4>, " << ibeta_int_data.size() << "> ibeta_derivative_small_data = { {\n";
     

   for(unsigned i = 0; i < ibeta_int_data.size(); ++i)
   {
      mp_t a(ibeta_int_data[i][0]);
      mp_t b(ibeta_int_data[i][1]);
      mp_t x(ibeta_int_data[i][2]);

      std::cout << a << std::endl;
      std::cout << b << std::endl;
      std::cout << x << std::endl;

      mp_t bet = exp(boost::math::lgamma(a) + boost::math::lgamma(b) - boost::math::lgamma(a + b));
      mp_t d = pow(1 - x, b - 1) * pow(x, a - 1) / bet;

      ofs << "{{ SC_(" << a << "), SC_(" << b << "), SC_(" << x << "), SC_(" << d << ") }}\n";
   }

   ofs << "}};\n\n";


   return 0;
}


