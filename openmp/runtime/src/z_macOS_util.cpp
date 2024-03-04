/*
 * z_macOS_util.cpp -- platform specific routines.
 */

//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cmath>

#define _ISNANd std::isnan
#define _ISNANf std::isnan
#define _ISINFd std::isinf
#define _ISINFf std::isinf
#define _ISFINITEd std::isfinite
#define _ISFINITEf std::isfinite
#define _COPYSIGNd std::copysign
#define _COPYSIGNf std::copysign
#define _SCALBNd std::scalbn
#define _SCALBNf std::scalbn
#define _ABSd std::abs
#define _ABSf std::abs
#define _LOGBd std::logb
#define _LOGBf std::logb
#define _fmaxd std::fmax
#define _fmaxf std::fmax

extern "C" {

double _Complex __divdc3(double __a, double __b, double __c, double __d) {
  int __ilogbw = 0;
  // Can't use std::max, because that's defined in <algorithm>, and we don't
  // want to pull that in for every compile.  The CUDA headers define
  // ::max(float, float) and ::max(double, double), which is sufficient for us.
  double __logbw = _LOGBd(_fmaxd(_ABSd(__c), _ABSd(__d)));
  if (_ISFINITEd(__logbw)) {
    __ilogbw = (int)__logbw;
    __c = _SCALBNd(__c, -__ilogbw);
    __d = _SCALBNd(__d, -__ilogbw);
  }
  double __denom = __c * __c + __d * __d;
  double _Complex z;
  __real__(z) = _SCALBNd((__a * __c + __b * __d) / __denom, -__ilogbw);
  __imag__(z) = _SCALBNd((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (_ISNANd(__real__(z)) && _ISNANd(__imag__(z))) {
    if ((__denom == 0.0) && (!_ISNANd(__a) || !_ISNANd(__b))) {
      __real__(z) = _COPYSIGNd(__builtin_huge_val(), __c) * __a;
      __imag__(z) = _COPYSIGNd(__builtin_huge_val(), __c) * __b;
    } else if ((_ISINFd(__a) || _ISINFd(__b)) && _ISFINITEd(__c) &&
               _ISFINITEd(__d)) {
      __a = _COPYSIGNd(_ISINFd(__a) ? 1.0 : 0.0, __a);
      __b = _COPYSIGNd(_ISINFd(__b) ? 1.0 : 0.0, __b);
      __real__(z) = __builtin_huge_val() * (__a * __c + __b * __d);
      __imag__(z) = __builtin_huge_val() * (__b * __c - __a * __d);
    } else if (_ISINFd(__logbw) && __logbw > 0.0 && _ISFINITEd(__a) &&
               _ISFINITEd(__b)) {
      __c = _COPYSIGNd(_ISINFd(__c) ? 1.0 : 0.0, __c);
      __d = _COPYSIGNd(_ISINFd(__d) ? 1.0 : 0.0, __d);
      __real__(z) = 0.0 * (__a * __c + __b * __d);
      __imag__(z) = 0.0 * (__b * __c - __a * __d);
    }
  }
  return z;
}

float _Complex __divsc3(float __a, float __b, float __c, float __d) {
  int __ilogbw = 0;
  float __logbw = _LOGBf(_fmaxf(_ABSf(__c), _ABSf(__d)));
  if (_ISFINITEf(__logbw)) {
    __ilogbw = (int)__logbw;
    __c = _SCALBNf(__c, -__ilogbw);
    __d = _SCALBNf(__d, -__ilogbw);
  }
  float __denom = __c * __c + __d * __d;
  float _Complex z;
  __real__(z) = _SCALBNf((__a * __c + __b * __d) / __denom, -__ilogbw);
  __imag__(z) = _SCALBNf((__b * __c - __a * __d) / __denom, -__ilogbw);
  if (_ISNANf(__real__(z)) && _ISNANf(__imag__(z))) {
    if ((__denom == 0) && (!_ISNANf(__a) || !_ISNANf(__b))) {
      __real__(z) = _COPYSIGNf(__builtin_huge_valf(), __c) * __a;
      __imag__(z) = _COPYSIGNf(__builtin_huge_valf(), __c) * __b;
    } else if ((_ISINFf(__a) || _ISINFf(__b)) && _ISFINITEf(__c) &&
               _ISFINITEf(__d)) {
      __a = _COPYSIGNf(_ISINFf(__a) ? 1 : 0, __a);
      __b = _COPYSIGNf(_ISINFf(__b) ? 1 : 0, __b);
      __real__(z) = __builtin_huge_valf() * (__a * __c + __b * __d);
      __imag__(z) = __builtin_huge_valf() * (__b * __c - __a * __d);
    } else if (_ISINFf(__logbw) && __logbw > 0 && _ISFINITEf(__a) &&
               _ISFINITEf(__b)) {
      __c = _COPYSIGNf(_ISINFf(__c) ? 1 : 0, __c);
      __d = _COPYSIGNf(_ISINFf(__d) ? 1 : 0, __d);
      __real__(z) = 0 * (__a * __c + __b * __d);
      __imag__(z) = 0 * (__b * __c - __a * __d);
    }
  }
  return z;
}
}
