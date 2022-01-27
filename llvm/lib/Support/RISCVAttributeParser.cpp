//===-- RISCVAttributeParser.cpp - RISCV Attribute Parser -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/RISCVAttributeParser.h"
#include "llvm/ADT/StringExtras.h"

using namespace llvm;

const RISCVAttributeParser::DisplayHandler
    RISCVAttributeParser::displayRoutines[] = {
        {
            RISCVAttrs::ARCH,
            &ELFAttributeParser::stringAttribute,
        },
        {
            RISCVAttrs::PRIV_SPEC,
            &ELFAttributeParser::integerAttribute,
        },
        {
            RISCVAttrs::PRIV_SPEC_MINOR,
            &ELFAttributeParser::integerAttribute,
        },
        {
            RISCVAttrs::PRIV_SPEC_REVISION,
            &ELFAttributeParser::integerAttribute,
        },
        {
            RISCVAttrs::STACK_ALIGN,
            &RISCVAttributeParser::stackAlign,
        },
        {
            RISCVAttrs::UNALIGNED_ACCESS,
            &RISCVAttributeParser::unalignedAccess,
        }};

Error RISCVAttributeParser::unalignedAccess(unsigned tag) {
  static const char *strings[] = {"No unaligned access", "Unaligned access"};
  return parseStringAttribute("Unaligned_access", tag, makeArrayRef(strings));
}

Error RISCVAttributeParser::stackAlign(unsigned tag) {
  uint64_t value = de.getULEB128(cursor);
  std::string description =
      "Stack alignment is " + utostr(value) + std::string("-bytes");
  printAttribute(tag, value, description);
  return Error::success();
}

Error RISCVAttributeParser::handler(uint64_t tag, bool &handled) {
  handled = false;
  for (unsigned AHI = 0, AHE = array_lengthof(displayRoutines); AHI != AHE;
       ++AHI) {
    if (uint64_t(displayRoutines[AHI].attribute) == tag) {
      if (Error e = (this->*displayRoutines[AHI].routine)(tag))
        return e;
      handled = true;
      break;
    }
  }

  return Error::success();
}
