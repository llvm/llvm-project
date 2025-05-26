//  (C) Copyright John Maddock 2007.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/limits.hpp>
#include <vector>
#include "mp_t.hpp"

void write_table(unsigned max_exponent)
{
   mp_t max = ldexp(mp_t(1), (int)max_exponent);

   std::vector<mp_t> factorials;
   factorials.push_back(1);

   mp_t f(1);
   unsigned i = 1;

   while(f < max)
   {
      factorials.push_back(f);
      ++i;
      f *= i;
   }

   //
   // now write out the results to cout:
   //
   std::cout << std::scientific << std::setprecision(40);
   std::cout << "   static const std::array<T, " << factorials.size() << "> factorials = {\n";
   for(unsigned j = 0; j < factorials.size(); ++j)
      std::cout << "      " << factorials[j] << "L,\n";
   std::cout << "   };\n\n";
}


int main()
{
   write_table(16384/*std::numeric_limits<float>::max_exponent*/);
}
