//===-- IsInf.cpp - Platform-independent wrapper around C99 isinf() -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Platform-independent wrapper around C99 isinf()
//
//===----------------------------------------------------------------------===//

#include "llvm/Config/config.h"

#if HAVE_ISINF_IN_MATH_H
# include <math.h>
#elif HAVE_ISINF_IN_CMATH
# include <cmath>
#elif HAVE_STD_ISINF_IN_CMATH
# include <cmath>
using std::isinf;
#elif HAVE_FINITE_IN_IEEEFP_H
// A handy workaround I found at http://www.unixguide.net/sun/faq ...
// apparently this has been a problem with Solaris for years.
# include <ieeefp.h>
static int isinf(double x) { return !finite(x) && x==x; }
#elif defined(_MSC_VER)
#include <float.h>
#define isinf(X) (!_finite(X))
#elif defined(_AIX) && defined(__GNUC__)
// GCC's fixincludes seems to be removing the isinf() declaration from the
// system header /usr/include/math.h
# include <math.h>
static int isinf(double x) { return !finite(x) && x==x; }
#elif defined(__hpux)
// HP-UX is "special"
#include <math.h>
static int isinf(double x) { return ((x) == INFINITY) || ((x) == -INFINITY); }
#else
# error "Don't know how to get isinf()"
#endif

namespace llvm {

int IsInf(float f)  { return isinf(f); }
int IsInf(double d) { return isinf(d); }

} // end namespace llvm;
