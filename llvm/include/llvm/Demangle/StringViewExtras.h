//===--- StringViewExtras.h ----------*- mode:c++;eval:(read-only-mode) -*-===//
//       Do not edit! See README.txt.
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// There are two copies of this file in the source tree.  The one under
// libcxxabi is the original and the one under llvm is the copy.  Use
// cp-to-llvm.sh to update the copy.  See README.txt for more details.
//
//===----------------------------------------------------------------------===//

#ifndef DEMANGLE_STRINGVIEW_H
#define DEMANGLE_STRINGVIEW_H

#include "DemangleConfig.h"

#include <string_view>

DEMANGLE_NAMESPACE_BEGIN

inline bool starts_with(std::string_view self, char C) {
  return !self.empty() && self.front() == C;
}

inline bool starts_with(std::string_view haystack, std::string_view needle) {
  return haystack.substr(0, needle.size()) == needle;
}

DEMANGLE_NAMESPACE_END

#endif
