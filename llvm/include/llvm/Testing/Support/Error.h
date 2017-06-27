//===- llvm/Testing/Support/Error.h ---------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TESTING_SUPPORT_ERROR_H
#define LLVM_TESTING_SUPPORT_ERROR_H

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/Testing/Support/SupportHelpers.h"

#include "gmock/gmock.h"
#include <ostream>

namespace llvm {
namespace detail {
ErrorHolder TakeError(Error Err);

template <typename T> ExpectedHolder<T> TakeExpected(Expected<T> &Exp) {
  llvm::detail::ExpectedHolder<T> Result;
  auto &EH = static_cast<llvm::detail::ErrorHolder &>(Result);
  EH = TakeError(Exp.takeError());
  if (Result.Success)
    Result.Value = &(*Exp);
  return Result;
}

template <typename T> ExpectedHolder<T> TakeExpected(Expected<T> &&Exp) {
  return TakeExpected(Exp);
}
} // namespace detail

#define EXPECT_THAT_ERROR(Err, Matcher)                                        \
  EXPECT_THAT(llvm::detail::TakeError(Err), Matcher)
#define ASSERT_THAT_ERROR(Err, Matcher)                                        \
  ASSERT_THAT(llvm::detail::TakeError(Err), Matcher)

#define EXPECT_THAT_EXPECTED(Err, Matcher)                                     \
  EXPECT_THAT(llvm::detail::TakeExpected(Err), Matcher)
#define ASSERT_THAT_EXPECTED(Err, Matcher)                                     \
  ASSERT_THAT(llvm::detail::TakeExpected(Err), Matcher)

MATCHER(Succeeded, "") { return arg.Success; }
MATCHER(Failed, "") { return !arg.Success; }

MATCHER_P(HasValue, value,
          "succeeded with value " + testing::PrintToString(value)) {
  if (!arg.Success) {
    *result_listener << "operation failed";
    return false;
  }

  assert(arg.Value.hasValue());
  if (**arg.Value != value) {
    *result_listener << "but \"" + testing::PrintToString(**arg.Value) +
                            "\" != " + testing::PrintToString(value);
    return false;
  }

  return true;
}
} // namespace llvm

#endif
