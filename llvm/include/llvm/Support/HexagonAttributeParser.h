//===-- HexagonAttributeParser.h - Hexagon Attribute Parser -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_HEXAGONATTRIBUTEPARSER_H
#define LLVM_SUPPORT_HEXAGONATTRIBUTEPARSER_H

#include "llvm/Support/Compiler.h"
#include "llvm/Support/ELFAttrParserCompact.h"
#include "llvm/Support/HexagonAttributes.h"

namespace llvm {
class LLVM_ABI HexagonAttributeParser : public ELFCompactAttrParser {
  struct DisplayHandler {
    HexagonAttrs::AttrType Attribute;
    Error (HexagonAttributeParser::*Routine)(unsigned);
  };

  static const DisplayHandler DisplayRoutines[];

  Error handler(uint64_t Tag, bool &Handled) override;

public:
  HexagonAttributeParser(ScopedPrinter *SP)
      : ELFCompactAttrParser(SP, HexagonAttrs::getHexagonAttributeTags(),
                             "hexagon") {}
  HexagonAttributeParser()
      : ELFCompactAttrParser(HexagonAttrs::getHexagonAttributeTags(),
                             "hexagon") {}
};

} // namespace llvm

#endif
