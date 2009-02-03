//===-- IsNAN.cpp ---------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Platform-independent wrapper around C99 isnan().
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"

#if HAVE_ISNAN_IN_MATH_H
# include <math.h>
#elif HAVE_ISNAN_IN_CMATH
# include <cmath>
#elif HAVE_STD_ISNAN_IN_CMATH
# include <cmath>
using std::isnan;
#elif defined(_MSC_VER)
#include <float.h>
#define isnan _isnan
#else
# error "Don't know how to get isnan()"
#endif

namespace llvm {
  int IsNAN(float f)  { return isnan(f); }
  int IsNAN(double d) { return isnan(d); }
} // end namespace llvm;
