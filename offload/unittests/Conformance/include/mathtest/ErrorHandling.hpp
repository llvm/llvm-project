//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the definition of error handling macros for reporting
/// fatal error conditions and validating Offload API calls.
///
//===----------------------------------------------------------------------===//

#ifndef MATHTEST_ERRORHANDLING_HPP
#define MATHTEST_ERRORHANDLING_HPP

#include "mathtest/OffloadForward.hpp"

#include "llvm/ADT/Twine.h"

#define FATAL_ERROR(Message)                                                   \
  mathtest::detail::reportFatalError(Message, __FILE__, __LINE__, __func__)

#define OL_CHECK(ResultExpr)                                                   \
  do {                                                                         \
    ol_result_t Result = (ResultExpr);                                         \
    if (Result != OL_SUCCESS)                                                  \
      mathtest::detail::reportOffloadError(#ResultExpr, Result, __FILE__,      \
                                           __LINE__, __func__);                \
                                                                               \
  } while (false)

namespace mathtest {
namespace detail {

[[noreturn]] void reportFatalError(const llvm::Twine &Message, const char *File,
                                   int Line, const char *FuncName);

[[noreturn]] void reportOffloadError(const char *ResultExpr, ol_result_t Result,
                                     const char *File, int Line,
                                     const char *FuncName);
} // namespace detail
} // namespace mathtest

#endif // MATHTEST_ERRORHANDLING_HPP
