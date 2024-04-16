//===- offload-tblgen/GenCommon.cpp - Common defs for Offload generators --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "RecordTypes.hpp"
#include "llvm/Support/FormatVariadic.h"

constexpr auto CommentsHeader = R"(
///////////////////////////////////////////////////////////////////////////////
)";

constexpr auto CommentsBreak = "///\n";

constexpr auto PrefixLower = "ol";
constexpr auto PrefixUpper = "OL";

static std::string
MakeParamComment(const llvm::offload::tblgen::ParamRec &Param) {
  return llvm::formatv("///< {0}{1}{2} {3}", (Param.isIn() ? "[in]" : ""),
                       (Param.isOut() ? "[out]" : ""),
                       (Param.isOpt() ? "[optional]" : ""), Param.getDesc());
}
