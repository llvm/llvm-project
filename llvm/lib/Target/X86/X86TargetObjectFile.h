//===-- X86TargetObjectFile.h - X86 Object Info -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_X86_TARGETOBJECTFILE_H
#define LLVM_TARGET_X86_TARGETOBJECTFILE_H

#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {

  /// X86_64MachoTargetObjectFile - This TLOF implementation is used for Darwin
  /// x86-64.
  class X86_64MachoTargetObjectFile : public TargetLoweringObjectFileMachO {
  public:
    virtual const MCExpr *
    getExprForDwarfGlobalReference(const GlobalValue *GV, Mangler *Mang,
                                   MachineModuleInfo *MMI, unsigned Encoding,
                                   MCStreamer &Streamer) const;

    // getCFIPersonalitySymbol - The symbol that gets passed to
    // .cfi_personality.
    virtual MCSymbol *
    getCFIPersonalitySymbol(const GlobalValue *GV, Mangler *Mang,
                            MachineModuleInfo *MMI) const;
  };

  /// X86LinuxTargetObjectFile - This implementation is used for linux x86
  /// and x86-64.
  class X86LinuxTargetObjectFile : public TargetLoweringObjectFileELF {
    virtual void Initialize(MCContext &Ctx, const TargetMachine &TM);
  };

} // end namespace llvm

#endif
