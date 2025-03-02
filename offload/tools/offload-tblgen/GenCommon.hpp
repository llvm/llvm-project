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

// Having inline bits of tabbed code is hard to read, provide some definitions
// so we can keep things tidier
#define TAB_1 "  "
#define TAB_2 "    "
#define TAB_3 "      "
#define TAB_4 "        "
#define TAB_5 "          "

constexpr auto GenericHeader =
    R"(//===- Auto-generated file, part of the LLVM/Offload project --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
)";

constexpr auto FileHeader = R"(
// Auto-generated file, do not manually edit.

#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

)";

constexpr auto FileFooter = R"(
#if defined(__cplusplus)
} // extern "C"
#endif

)";

constexpr auto CommentsHeader = R"(
///////////////////////////////////////////////////////////////////////////////
)";

constexpr auto CommentsBreak = "///\n";

constexpr auto PrefixLower = "ol";
constexpr auto PrefixUpper = "OL";

inline std::string
MakeParamComment(const llvm::offload::tblgen::ParamRec &Param) {
  return llvm::formatv("// {0}{1}{2} {3}", (Param.isIn() ? "[in]" : ""),
                       (Param.isOut() ? "[out]" : ""),
                       (Param.isOpt() ? "[optional]" : ""), Param.getDesc());
}
