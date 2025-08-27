//===- OmptTester.h - Main header for ompTest usage -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file represents the main header file for usage of the ompTest library.
/// Depending on the build either 'standalone' or GoogleTest headers are
/// included and corresponding main-function macros are defined.
///
//===----------------------------------------------------------------------===//

#ifndef OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTTESTER_H
#define OPENMP_TOOLS_OMPTEST_INCLUDE_OMPTTESTER_H

#include "AssertMacros.h"
#include "Logging.h"
#include "OmptAliases.h"
#include "OmptAssertEvent.h"
#include "OmptAsserter.h"
#include "OmptCallbackHandler.h"

#include <cassert>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

// Standalone header section
#ifdef OPENMP_LIBOMPTEST_BUILD_STANDALONE

#include "OmptTesterStandalone.h"

// Define standalone main function (place once at the bottom of a testsuite)
#define OMPTEST_TESTSUITE_MAIN()                                               \
  int main(int argc, char **argv) {                                            \
    Runner R;                                                                  \
    return R.run();                                                            \
  }

// GoogleTest header section
#else

#include "OmptTesterGoogleTest.h"

// Define GoogleTest main function (place once at the bottom of a testsuite)
#define OMPTEST_TESTSUITE_MAIN()                                               \
  int main(int argc, char **argv) {                                            \
    testing::InitGoogleTest(&argc, argv);                                      \
    return RUN_ALL_TESTS();                                                    \
  }

#endif

#endif
