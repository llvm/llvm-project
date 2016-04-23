//===-- llvm/Target/ARMTargetObjectFile.h - ARM Object Info -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMTARGETOBJECTFILE_H
#define LLVM_LIB_TARGET_ARM_ARMTARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"

namespace llvm {

class MCContext;
class TargetMachine;

class ARMElfTargetObjectFile : public TargetLoweringObjectFileELF {
protected:
  const MCSection *AttributesSection;
public:
  ARMElfTargetObjectFile()
      : TargetLoweringObjectFileELF(), AttributesSection(nullptr) {
    PLTRelativeVariantKind = MCSymbolRefExpr::VK_ARM_PREL31;
  }

  void Initialize(MCContext &Ctx, const TargetMachine &TM) override;

  const MCExpr *
  getTTypeGlobalReference(const GlobalValue *GV, unsigned Encoding,
                          Mangler &Mang, const TargetMachine &TM,
                          MachineModuleInfo *MMI,
                          MCStreamer &Streamer) const override;

  /// \brief Describe a TLS variable address within debug info.
  const MCExpr *getDebugThreadLocalSymbol(const MCSymbol *Sym) const override;
};

} // end namespace llvm

#endif
