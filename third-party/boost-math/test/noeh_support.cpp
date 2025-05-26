//  Copyright John Maddock 2015.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/config.hpp>
#include <iostream>
#include <iomanip>
#include <cstdlib>


#ifdef BOOST_NO_EXCEPTIONS

namespace boost {

   void throw_exception(const std::exception& e)
   {
      std::cout << e.what() << std::endl;
      std::abort();
   }


}

#endif
