//===-- DWARFAttribute.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_DWARFATTRIBUTE_H
#define LLVM_DEBUGINFO_DWARFATTRIBUTE_H

#include "llvm/Support/DataTypes.h"

namespace llvm {

class DWARFAttribute {
  uint16_t Attribute;
  uint16_t Form;
  public:
  DWARFAttribute(uint16_t attr, uint16_t form)
    : Attribute(attr), Form(form) {}

  uint16_t getAttribute() const { return Attribute; }
  uint16_t getForm() const { return Form; }
};

}

#endif
