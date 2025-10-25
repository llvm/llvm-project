//===---- CSKYAttributeParser.h - CSKY Attribute Parser ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_CSKYATTRIBUTEPARSER_H
#define LLVM_SUPPORT_CSKYATTRIBUTEPARSER_H

#include "llvm/Support/CSKYAttributes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ELFAttrParserCompact.h"

namespace llvm {
class LLVM_ABI CSKYAttributeParser : public ELFCompactAttrParser {
  struct DisplayHandler {
    CSKYAttrs::AttrType attribute;
    Error (CSKYAttributeParser::*routine)(unsigned);
  };
  static const DisplayHandler displayRoutines[];

  Error dspVersion(unsigned tag);
  Error vdspVersion(unsigned tag);
  Error fpuVersion(unsigned tag);
  Error fpuABI(unsigned tag);
  Error fpuRounding(unsigned tag);
  Error fpuDenormal(unsigned tag);
  Error fpuException(unsigned tag);
  Error fpuHardFP(unsigned tag);

  Error handler(uint64_t tag, bool &handled) override;

public:
  CSKYAttributeParser(ScopedPrinter *sw)
      : ELFCompactAttrParser(sw, CSKYAttrs::getCSKYAttributeTags(), "csky") {}
  CSKYAttributeParser()
      : ELFCompactAttrParser(CSKYAttrs::getCSKYAttributeTags(), "csky") {}
};

} // namespace llvm

#endif
