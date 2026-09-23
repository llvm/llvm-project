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

#if defined(_WIN32)
#if defined(RTTI_CROSS_DYLIB_TEST_LIB_EXPORTS)
#define RTTI_TEST_EXPORT __declspec(dllexport)
#else
#define RTTI_TEST_EXPORT __declspec(dllimport)
#endif
#else
#define RTTI_TEST_EXPORT __attribute__((visibility("default")))
#endif

extern "C" RTTI_TEST_EXPORT orc_rt::ErrorInfoBase *
rttiCrossDylibTest_makeError(int Code);

extern "C" RTTI_TEST_EXPORT void
rttiCrossDylibTest_destroyError(orc_rt::ErrorInfoBase *E);

extern "C" RTTI_TEST_EXPORT const void *rttiCrossDylibTest_libraryID();

#endif // ORC_RT_TEST_RTTICROSSDYLIBTESTLIB_H
