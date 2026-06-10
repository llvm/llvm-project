//===- SystemZGOFFObjectWriter.cpp - SystemZ GOFF writer ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZMCAsmInfo.h"
#include "llvm/MC/MCGOFFObjectWriter.h"
#include <memory>

using namespace llvm;

namespace {
class SystemZGOFFObjectWriter : public MCGOFFObjectTargetWriter {
public:
  SystemZGOFFObjectWriter();

  unsigned getRelocType(const MCValue &Target,
                        const MCFixup &Fixup) const override;
};
} // end anonymous namespace

SystemZGOFFObjectWriter::SystemZGOFFObjectWriter()
    : MCGOFFObjectTargetWriter() {}

unsigned SystemZGOFFObjectWriter::getRelocType(const MCValue &Target,
                                               const MCFixup &Fixup) const {
  switch (Target.getSpecifier()) {
  case SystemZ::S_RCon:
    return Reloc_Type_RCon;
  case SystemZ::S_VCon:
    return Reloc_Type_VCon;
  case SystemZ::S_QCon:
    return Reloc_Type_QCon;
  case SystemZ::S_None:
    if (Fixup.isPCRel())
      return Reloc_Type_RICon;
    return Reloc_Type_ACon;
  }
  llvm_unreachable("Modifier not supported");
}

std::unique_ptr<MCObjectTargetWriter> llvm::createSystemZGOFFObjectWriter() {
  return std::make_unique<SystemZGOFFObjectWriter>();
}
