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
// load/store pseudos. The movrel form (V_LOAD_IDX_B<N> / V_STORE_IDX_B<N>)
// reads its index from M0; the VGPR indexing mode form
// (V_LOAD_IDX_GPR_IDX_B<N> / V_STORE_IDX_GPR_IDX_B<N>) takes it in an SGPR:
//   movrel:   load (outs data), (ins offset)      store (ins data, offset)
//   gpr_idx:  load (outs data), (ins idx, offset) store (ins data, idx, offset)
class VLoadStoreIdxInst : public MachineInstr {
public:
  bool isGPRIdx() const;

  MachineOperand &getDataOp() { return getOperand(0); }
  MachineOperand &getIdxOp() {
    assert(isGPRIdx() && "movrel form reads its index from M0");
    return getOperand(1);
  }
  MachineOperand &getOffsetOp() { return getOperand(isGPRIdx() ? 2 : 1); }
  const MachineOperand &getDataOp() const { return getOperand(0); }
  const MachineOperand &getIdxOp() const {
    assert(isGPRIdx() && "movrel form reads its index from M0");
    return getOperand(1);
  }
  const MachineOperand &getOffsetOp() const {
    return getOperand(isGPRIdx() ? 2 : 1);
  }

  unsigned getBitWidth() const;

  static bool classof(const MachineInstr *MI) {
    return AMDGPU::getVLdStIdxOpcodeInfoByOpcode(MI->getOpcode()) != nullptr;
  }
};

class VLoadIdxInst : public VLoadStoreIdxInst {
public:
  static int tryGetOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx = false);
  static unsigned getOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx = false);

  static bool classof(const MachineInstr *MI) {
    const AMDGPU::VLdStIdxOpcodeInfo *Info =
        AMDGPU::getVLdStIdxOpcodeInfoByOpcode(MI->getOpcode());
    return Info && !Info->IsStore;
  }
};

class VStoreIdxInst : public VLoadStoreIdxInst {
public:
  static int tryGetOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx = false);
  static unsigned getOpcodeForBitWidth(unsigned Bits, bool IsGPRIdx = false);

  static bool classof(const MachineInstr *MI) {
    const AMDGPU::VLdStIdxOpcodeInfo *Info =
        AMDGPU::getVLdStIdxOpcodeInfoByOpcode(MI->getOpcode());
    return Info && Info->IsStore;
  }
};

} // end namespace AMDGPUMI
} // end namespace llvm

#endif // LLVM_LIB_TARGET_AMDGPU_AMDGPUMACHINEINSTRS_H
