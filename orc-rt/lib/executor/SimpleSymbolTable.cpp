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

Error SimpleSymbolTable::makeIncompatibleDefsError(
    std::vector<std::string_view> IncompatibleDefs) {
  std::sort(IncompatibleDefs.begin(), IncompatibleDefs.end());
  std::string ErrMsg = "Incompatible definitions for symbols: [ ";
  ErrMsg += IncompatibleDefs.front();
  for (auto &Def : iterator_range(std::next(IncompatibleDefs.begin()),
                                  IncompatibleDefs.end())) {
    ErrMsg += ", ";
    ErrMsg += Def;
  }
  ErrMsg += " ]";
  return make_error<StringError>(std::move(ErrMsg));
}

} // namespace orc_rt
