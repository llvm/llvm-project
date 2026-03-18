//===- SimpleSymbolTable.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/SimpleSymbolTable.h
// header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/SimpleSymbolTable.h"
#include "orc-rt/iterator_range.h"

#include <algorithm>

namespace orc_rt {

Error SimpleSymbolTable::makeDuplicatesError(
    std::vector<std::string_view> Dups) {
  std::sort(Dups.begin(), Dups.end());
  std::string ErrMsg = "Could not add duplicate symbols: [ ";
  ErrMsg += Dups.front();
  for (auto &Dup : iterator_range(std::next(Dups.begin()), Dups.end())) {
    ErrMsg += ", ";
    ErrMsg += Dup;
  }
  ErrMsg += " ]";
  return make_error<StringError>(std::move(ErrMsg));
}

} // namespace orc_rt
