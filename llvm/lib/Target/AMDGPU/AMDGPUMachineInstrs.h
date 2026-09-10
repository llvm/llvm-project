//===-- AMDGPUMachineInstrs.h -*- C++ -*-----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// Convenience wrappers and helpers for AMDGPU-specific machine instructions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H

#include "SIInstrInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MachineInstr.h"

namespace llvm {
namespace AMDGPUMI {

// Wrapper for the whole-dword VGPR "as memory" (address space 13) indexed
// load/store pseudos (V_LOAD_IDX_B<N> / V_STORE_IDX_B<N>). Operand layout:
//   load:  (outs data), (ins idx, offset)
//   store: (outs),      (ins data, idx, offset)
// so data/idx/offset are always operands 0/1/2.
class VLoadStoreIdxInst : public MachineInstr {
public:
  MachineOperand &getDataOp() { return getOperand(0); }
  MachineOperand &getIdxOp() { return getOperand(1); }
  MachineOperand &getOffsetOp() { return getOperand(2); }
  const MachineOperand &getDataOp() const { return getOperand(0); }
  const MachineOperand &getIdxOp() const { return getOperand(1); }
  const MachineOperand &getOffsetOp() const { return getOperand(2); }

  unsigned getBitWidth() const;

  static bool classof(const MachineInstr *MI) {
    unsigned Opc = MI->getOpcode();
    if (Opc == AMDGPU::V_LOAD_IDX_BITS || Opc == AMDGPU::V_STORE_IDX_BITS)
      return true;
    return AMDGPU::getVLdStIdxOpcodeInfoByOpcode(Opc) != nullptr;
  }
};

class VLoadIdxInst : public VLoadStoreIdxInst {
public:
  static int tryGetOpcodeForBitWidth(unsigned Bits);
  static unsigned getOpcodeForBitWidth(unsigned Bits);

  static bool classof(const MachineInstr *MI) {
    unsigned Opc = MI->getOpcode();
    if (Opc == AMDGPU::V_LOAD_IDX_BITS)
      return true;
    const AMDGPU::VLdStIdxOpcodeInfo *Info =
        AMDGPU::getVLdStIdxOpcodeInfoByOpcode(Opc);
    return Info && !Info->IsStore;
  }
};

class VStoreIdxInst : public VLoadStoreIdxInst {
public:
  static int tryGetOpcodeForBitWidth(unsigned Bits);
  static unsigned getOpcodeForBitWidth(unsigned Bits);

  static bool classof(const MachineInstr *MI) {
    unsigned Opc = MI->getOpcode();
    if (Opc == AMDGPU::V_STORE_IDX_BITS)
      return true;
    const AMDGPU::VLdStIdxOpcodeInfo *Info =
        AMDGPU::getVLdStIdxOpcodeInfoByOpcode(Opc);
    return Info && Info->IsStore;
  }
};

} // end namespace AMDGPUMI
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H
