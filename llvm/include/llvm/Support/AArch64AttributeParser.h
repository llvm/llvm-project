//=== - AArch64AttributeParser.h-AArch64 Attribute Information Printer - ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_AARCH64ATTRIBUTEPARSER_H
#define LLVM_SUPPORT_AARCH64ATTRIBUTEPARSER_H

#include "ELFAttributeParser.h"
#include "llvm/Support/Error.h"

namespace llvm {

class ScopedPrinter;

class AArch64AttributeParser : public ELFAttributeParser {
  Error handler(uint64_t Tag, bool &Handled) override {
    return Error::success();
  }

public:
  Error parse(ArrayRef<uint8_t> Section, llvm::endianness Endian) override;

  AArch64AttributeParser(ScopedPrinter *Sw) : ELFAttributeParser(Sw) {}
  AArch64AttributeParser() {}
};
} // namespace llvm

#endif // LLVM_SUPPORT_AARCH64ATTRIBUTEPARSER_H
