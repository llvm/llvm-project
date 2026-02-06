//===- RemarkSerializer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides tools for serializing remarks.
//
//===----------------------------------------------------------------------===//

#include "llvm/Remarks/RemarkSerializer.h"
#include "llvm/Remarks/BitstreamRemarkSerializer.h"
#include "llvm/Remarks/YAMLRemarkSerializer.h"

using namespace llvm;
using namespace llvm::remarks;

Expected<std::unique_ptr<RemarkSerializer>>
remarks::createRemarkSerializer(Format RemarksFormat, raw_ostream &OS) {
  switch (RemarksFormat) {
  case Format::Unknown:
  case Format::Auto:
    return createStringError(std::errc::invalid_argument,
                             "Invalid remark serializer format.");
  case Format::YAML:
    return std::make_unique<YAMLRemarkSerializer>(OS);
  case Format::Bitstream:
    return std::make_unique<BitstreamRemarkSerializer>(OS);
  }
  llvm_unreachable("Unknown remarks::Format enum");
}

Expected<std::unique_ptr<RemarkSerializer>>
remarks::createRemarkSerializer(Format RemarksFormat, raw_ostream &OS,
                                remarks::StringTable StrTab) {
  switch (RemarksFormat) {
  case Format::Unknown:
  case Format::Auto:
    return createStringError(std::errc::invalid_argument,
                             "Invalid remark serializer format.");
  case Format::YAML:
    return std::make_unique<YAMLRemarkSerializer>(OS, std::move(StrTab));
  case Format::Bitstream:
    return std::make_unique<BitstreamRemarkSerializer>(OS, std::move(StrTab));
  }
  llvm_unreachable("Unknown remarks::Format enum");
}
