//===- Testing/Support/SupportHelpers.h -----------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TESTING_SUPPORT_SUPPORTHELPERS_H
#define LLVM_TESTING_SUPPORT_SUPPORTHELPERS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "gtest/gtest-printers.h"

namespace llvm {
namespace detail {
struct ErrorHolder {
  bool Success;
  std::string Message;
};

template <typename T> struct ExpectedHolder : public ErrorHolder {
  ExpectedHolder(ErrorHolder Err, Expected<T> &Exp)
      : ErrorHolder(std::move(Err)), Exp(Exp) {}

  Expected<T> &Exp;
};

inline void PrintTo(const ErrorHolder &Err, std::ostream *Out) {
  *Out << (Err.Success ? "succeeded" : "failed");
  if (!Err.Success) {
    *Out << "  (" << StringRef(Err.Message).trim().str() << ")";
  }
}

template <typename T>
void PrintTo(const ExpectedHolder<T> &Item, std::ostream *Out) {
  if (Item.Success) {
    *Out << "succeeded with value \"" << ::testing::PrintToString(*Item.Exp)
         << "\"";
  } else {
    PrintTo(static_cast<const ErrorHolder &>(Item), Out);
  }
}
} // namespace detail
} // namespace llvm

#endif
