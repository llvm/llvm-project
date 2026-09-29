//===-- SystemZELFAsmPrinter.h - SystemZ ELF asm printer --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the SystemZELFAsmPrinter class, which owns ELF-specific
// asm printer behaviour: ELF instruction lowering, fentry/stackmap/patchpoint,
// TLS block address, stack-guard address, and the ELF vector-ABI attribute.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZELFASMPRINTER_H
#define LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZELFASMPRINTER_H

#include "MCTargetDesc/SystemZTargetStreamer.h"
#include "SystemZAsmPrinter.h"
#include "SystemZMCInstLower.h"

namespace llvm {

class LLVM_LIBRARY_VISIBILITY SystemZELFAsmPrinter : public SystemZAsmPrinter {
  SystemZTargetStreamer *getTargetStreamer() {
    MCTargetStreamer *TS = OutStreamer->getTargetStreamer();
    assert(TS && "do not have a target streamer");
    return static_cast<SystemZTargetStreamer *>(TS);
  }

public:
  SystemZELFAsmPrinter(TargetMachine &TM, std::unique_ptr<MCStreamer> Streamer);

  void emitInstruction(const MachineInstr *MI) override;
  void emitMachineConstantPoolValue(MachineConstantPoolValue *MCPV) override;
  void emitEndOfAsmFile(Module &M) override;

private:
  void LowerFENTRY_CALL(const MachineInstr &MI, SystemZMCInstLower &Lower);
  void LowerSTACKMAP(const MachineInstr &MI);
  void LowerPATCHPOINT(const MachineInstr &MI, SystemZMCInstLower &Lower);
  void LowerPATCHABLE_FUNCTION_ENTER(const MachineInstr &MI,
                                     SystemZMCInstLower &Lower);
  void LowerPATCHABLE_RET(const MachineInstr &MI, SystemZMCInstLower &Lower);
  void lowerLOAD_TLS_BLOCK_ADDR(const MachineInstr &MI,
                                SystemZMCInstLower &Lower);
  void lowerLOAD_GLOBAL_STACKGUARD_ADDR(const MachineInstr &MI,
                                        SystemZMCInstLower &Lower);
  void emitAttributes(Module &M);
};

} // end namespace llvm

#endif // LLVM_LIB_TARGET_SYSTEMZ_SYSTEMZELFASMPRINTER_H
