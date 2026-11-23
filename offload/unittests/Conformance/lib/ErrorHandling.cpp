//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation of the helper functions for the error
/// handling macros.
///
//===----------------------------------------------------------------------===//

#include "mathtest/ErrorHandling.hpp"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <OffloadAPI.h>

using namespace mathtest;

[[noreturn]] void detail::reportFatalError(const llvm::Twine &Message,
                                           const char *File, int Line,
                                           const char *FuncName) {
  // clang-format off
  llvm::report_fatal_error(
      llvm::Twine("Fatal error in '") + FuncName +
          "' at " + File + ":" + llvm::Twine(Line) +
          "\n  Message: " + Message,
      /*gen_crash_diag=*/false);
  // clang-format on
}

[[noreturn]] void detail::reportOffloadError(const char *ResultExpr,
                                             ol_result_t Result,
                                             const char *File, int Line,
                                             const char *FuncName) {
  // clang-format off
  llvm::report_fatal_error(
      llvm::Twine("OL_CHECK failed") +
          "\n  Location: " + File + ":" + llvm::Twine(Line) +
          "\n  Function: " + FuncName +
          "\n  Expression: " + ResultExpr +
          "\n  Error code: " + llvm::Twine(Result->Code) +
          "\n  Details: " +
          (Result->Details ? Result->Details : "No details provided"),
      /*gen_crash_diag=*/false);
  // clang-format on
}
