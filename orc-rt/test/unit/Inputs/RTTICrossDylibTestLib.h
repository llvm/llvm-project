//===- RTTICrossDylibTestLib.h --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declarations for the functions exported by RTTICrossDylibTestLib, shared
// between the library's own definitions and RTTICrossDylibTest.cpp so the
// two agree on signatures at compile time.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_TEST_RTTICROSSDYLIBTESTLIB_H
#define ORC_RT_TEST_RTTICROSSDYLIBTESTLIB_H

#include "orc-rt/support/Error.h"

extern "C" orc_rt::ErrorInfoBase *rttiCrossDylibTest_makeError(int Code);
extern "C" void rttiCrossDylibTest_destroyError(orc_rt::ErrorInfoBase *E);
extern "C" const void *rttiCrossDylibTest_libraryID();

#endif // ORC_RT_TEST_RTTICROSSDYLIBTESTLIB_H
