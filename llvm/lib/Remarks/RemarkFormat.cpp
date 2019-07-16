//===- RemarkFormat.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of utilities to handle the different remark formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/RemarkFormat.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace llvm::remarks;

Expected<Format> llvm::remarks::parseFormat(StringRef FormatStr) {
  auto Result = StringSwitch<Format>(FormatStr)
                    .Cases("", "yaml", Format::YAML)
                    .Default(Format::Unknown);

  if (Result == Format::Unknown)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Unknown remark serializer format: '%s'",
                             FormatStr.data());

  return Result;
}
