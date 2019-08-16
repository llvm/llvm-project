//===-- RegularExpression.cpp -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/RegularExpression.h"

#include <string>

using namespace lldb_private;

RegularExpression::RegularExpression(llvm::StringRef str) { Compile(str); }

RegularExpression::RegularExpression(const RegularExpression &rhs)
    : RegularExpression() {
  Compile(rhs.GetText());
}

bool RegularExpression::Compile(llvm::StringRef str) {
  m_regex_text = str;
  m_regex = llvm::Regex(str);
  return IsValid();
}

bool RegularExpression::Execute(
    llvm::StringRef str,
    llvm::SmallVectorImpl<llvm::StringRef> *matches) const {
  return m_regex.match(str, matches);
}

bool RegularExpression::IsValid() const {
  std::string discarded;
  return m_regex.isValid(discarded);
}

llvm::StringRef RegularExpression::GetText() const { return m_regex_text; }

llvm::Error RegularExpression::GetError() const {
  std::string error;
  if (!m_regex.isValid(error))
    return llvm::make_error<llvm::StringError>(llvm::inconvertibleErrorCode(),
                                               error);
  return llvm::Error::success();
}
