//  Copyright John Maddock 2013.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_COMPILE_POISON_HPP
#define BOOST_MATH_COMPILE_POISON_HPP

#include <cmath>
#include <math.h>

//
// As per https://github.com/boostorg/math/issues/126
// we basically need to include every std lib header we use, otherwise
// our poisoned macros can break legit std lib code.
//
#include <valarray>
#include <complex>
#include <iosfwd>
#include <sstream>
#include <ostream>
#include <istream>
#include <utility>
#include <iomanip>
#include <typeinfo>
#include <stdexcept>
#include <cstddef>
#include <string>
#include <cstring>
#include <cctype>
#include <limits>
#include <exception>
#include <iterator>
#include <numeric>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <memory>
#include <cerrno>
#include <functional>
#include <future>
#include <thread>
#include <random>
#include <chrono>
#include <map>

//
// We have to include this *before* poisoning the macros
// as it needs to be able to use them!
//
#include <boost/math/special_functions/fpclassify.hpp>
//
// lexical_cast uses macro unsafe isinf etc, so we have to include this as well:
//
#ifndef BOOST_MATH_STANDALONE
#include <boost/lexical_cast.hpp>
#endif

//
// Poison all the function-like macros in C99 so if we accidentally call them
// in an unsafe manner, we'll get compiler errors.  Of course these shouldn't be
// macros in C++ at all...
//

#ifdef fpclassify
#undef fpclassify
#endif

#define fpclassify(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isfinite
#undef isfinite
#endif

#define isfinite(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isinf
#undef isinf
#endif

#define isinf(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isnan
#undef isnan
#endif

#define isnan(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isnormal
#undef isnormal
#endif

#define isnormal(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef signbit
#undef signbit
#endif

#define signbit(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isgreater
#undef isgreater
#endif

#define isgreater(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isgreaterequal
#undef isgreaterequal
#endif

#define isgreaterequal(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isless
#undef isless
#endif

#define isless(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef islessequal
#undef islessequal
#endif

#define islessequal(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef islessgreater
#undef islessgreater
#endif

#define islessgreater(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#ifdef isunordered
#undef isunordered
#endif

#define isunordered(x) this_should_not_compile(x)}}}}}}}}}}}}}}}}}}}

#endif

