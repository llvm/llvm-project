//===- SimpleSymbolTable.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the implementation of APIs in the orc-rt/bedrock/SimpleSymbolTable.h
// header.
//
//===----------------------------------------------------------------------===//

#include "orc-rt/bedrock/SimpleSymbolTable.h"
#include "orc-rt-internal/support/StringExtras.h"
#include "orc-rt/support/iterator_range.h"

#include <algorithm>

namespace orc_rt {

Error SimpleSymbolTable::makeIncompatibleDefsError(
    std::vector<std::string_view> IncompatibleDefs) {
  std::sort(IncompatibleDefs.begin(), IncompatibleDefs.end());
  return make_error<StringError>((StringOutputStream()
                                  << "incompatible definitions for symbols: [ "
                                  << join(IncompatibleDefs, ", ") << " ]")
                                     .str());
}

} // namespace orc_rt
