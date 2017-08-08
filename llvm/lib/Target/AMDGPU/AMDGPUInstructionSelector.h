//===- AMDGPUInstructionSelector --------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file declares the targeting of the InstructionSelector class for
/// AMDGPU.
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRUCTIONSELECTOR_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUINSTRUCTIONSELECTOR_H

#include "AMDGPU.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"

namespace llvm {

class AMDGPUInstrInfo;
class AMDGPURegisterBankInfo;
class MachineInstr;
class MachineOperand;
class MachineRegisterInfo;
class SIInstrInfo;
class SIRegisterInfo;
class SISubtarget;

class AMDGPUInstructionSelector : public InstructionSelector {
public:
  AMDGPUInstructionSelector(const SISubtarget &STI,
                            const AMDGPURegisterBankInfo &RBI);

  bool select(MachineInstr &I) const override;
private:
  struct GEPInfo {
    const MachineInstr &GEP;
    SmallVector<unsigned, 2> SgprParts;
    SmallVector<unsigned, 2> VgprParts;
    int64_t Imm;
    GEPInfo(const MachineInstr &GEP) : GEP(GEP), Imm(0) { }
  };

  MachineOperand getSubOperand64(MachineOperand &MO, unsigned SubIdx) const;
  bool selectG_CONSTANT(MachineInstr &I) const;
  bool selectG_ADD(MachineInstr &I) const;
  bool selectG_GEP(MachineInstr &I) const;
  bool hasVgprParts(ArrayRef<GEPInfo> AddrInfo) const;
  void getAddrModeInfo(const MachineInstr &Load, const MachineRegisterInfo &MRI,
                       SmallVectorImpl<GEPInfo> &AddrInfo) const;
  bool selectSMRD(MachineInstr &I, ArrayRef<GEPInfo> AddrInfo) const;
  bool selectG_LOAD(MachineInstr &I) const;
  bool selectG_STORE(MachineInstr &I) const;

  const SIInstrInfo &TII;
  const SIRegisterInfo &TRI;
  const AMDGPURegisterBankInfo &RBI;
protected:
  AMDGPUAS AMDGPUASI;
};

} // End llvm namespace.
#endif
