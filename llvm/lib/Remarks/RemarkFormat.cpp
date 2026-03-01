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
#include "llvm/Remarks/BitstreamRemarkContainer.h"

using namespace llvm;
using namespace llvm::remarks;

Expected<Format> llvm::remarks::parseFormat(StringRef FormatStr) {
  auto Result = StringSwitch<Format>(FormatStr)
                    .Cases({"", "yaml"}, Format::YAML)
                    .Case("bitstream", Format::Bitstream)
                    .Default(Format::Unknown);

  if (Result == Format::Unknown)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Unknown remark format: '%s'",
                             FormatStr.data());

  return Result;
}

Expected<Format> llvm::remarks::magicToFormat(StringRef MagicStr) {
  auto Result =
      StringSwitch<Format>(MagicStr)
          .StartsWith("--- ", Format::YAML) // This is only an assumption.
          .StartsWith(remarks::Magic,
                      Format::YAML) // Needed for remark meta section
          .StartsWith(remarks::ContainerMagic, Format::Bitstream)
          .Default(Format::Unknown);

  if (Result == Format::Unknown)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Automatic detection of remark format failed. "
                             "Unknown magic number: '%.4s'",
                             MagicStr.data());
  return Result;
}

Expected<Format> llvm::remarks::detectFormat(Format Selected,
                                             StringRef MagicStr) {
  if (Selected == Format::Unknown)
    return createStringError(std::make_error_code(std::errc::invalid_argument),
                             "Unknown remark parser format.");
  if (Selected != Format::Auto)
    return Selected;

  // Empty files are valid bitstream files
  if (MagicStr.empty())
    return Format::Bitstream;
  return magicToFormat(MagicStr);
}
