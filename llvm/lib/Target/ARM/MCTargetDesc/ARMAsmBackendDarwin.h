//===-- ARMAsmBackendDarwin.h   ARM Asm Backend Darwin ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_ARM_ARMASMBACKENDDARWIN_H
#define LLVM_LIB_TARGET_ARM_ARMASMBACKENDDARWIN_H

#include "ARMAsmBackend.h"
#include "llvm/BinaryFormat/MachO.h"
#include "llvm/MC/MCObjectWriter.h"

namespace llvm {
class ARMAsmBackendDarwin : public ARMAsmBackend {
  const MCRegisterInfo &MRI;
public:
  const MachO::CPUSubTypeARM Subtype;
  ARMAsmBackendDarwin(const Target &T, const MCSubtargetInfo &STI,
                      const MCRegisterInfo &MRI, MachO::CPUSubTypeARM st)
      : ARMAsmBackend(T, STI, support::little), MRI(MRI), Subtype(st) {}

  std::unique_ptr<MCObjectTargetWriter>
  createObjectTargetWriter() const override {
    return createARMMachObjectWriter(/*Is64Bit=*/false, MachO::CPU_TYPE_ARM,
                                     Subtype);
  }

  uint32_t generateCompactUnwindEncoding(
      ArrayRef<MCCFIInstruction> Instrs) const override;
};
} // end namespace llvm

#endif
