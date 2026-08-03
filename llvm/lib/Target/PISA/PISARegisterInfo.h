//===-- PISARegisterInfo.h - PISA Register Information --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_PISA_PISAREGISTERINFO_H
#define LLVM_LIB_TARGET_PISA_PISAREGISTERINFO_H

#include "PISADefines.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "PISAGenRegisterInfo.inc"

namespace llvm {

class PISARegisterInfo : public PISAGenRegisterInfo {
public:
  PISARegisterInfo();
  const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
  BitVector getReservedRegs(const MachineFunction &MF) const override;
  bool eliminateFrameIndex(MachineBasicBlock::iterator MI, int SPAdj,
                           unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override {
    llvm_unreachable("unexpected execution");
  }
  Register getFrameRegister(const MachineFunction &MF) const override {
    return Register();
  }
  const TargetRegisterClass *getRegClassFromLLT(LLT Ty) const;
  unsigned getNumEltsFromRegClass(const TargetRegisterClass *RC) const;
  unsigned getBitSizeFromRegClass(const TargetRegisterClass *RC) const;
  const TargetRegisterClass *getVectorRegClass(unsigned NumElts,
                                               unsigned BitSize) const;
  unsigned getSubRegIdx(unsigned Size, unsigned Elt) const;

  // Return the composite sub-register index (.xy / .zw) covering `Count`
  // consecutive elements of `Size` bits starting at element `Base`, or 0 if
  // there is no nameable composite for that slice. Only 2-element pairs at
  // base 0 (.xy) or base 2 (.zw) are supported.
  unsigned getCompositeSubRegIdx(unsigned Size, unsigned Base,
                                 unsigned Count) const;

  PISA::Swizzle getSwizzle(unsigned SubReg) const;
  const char *getSwizzleName(unsigned SubReg) const;
  static bool isSelectorSwizzle(PISA::Swizzle Swizzle);

  bool isSpecialReg(Register Reg) const;

  bool shouldCoalesce(MachineInstr *MI, const TargetRegisterClass *SrcRC,
                      unsigned SubReg, const TargetRegisterClass *DstRC,
                      unsigned DstSubReg, const TargetRegisterClass *NewRC,
                      LiveIntervals &LIS) const override;

  const TargetRegisterClass *
  getMatchingSuperRegClass(const TargetRegisterClass *A,
                           const TargetRegisterClass *B,
                           unsigned SubIdx) const override;

private:
  unsigned getScalarBitSize(const TargetRegisterClass *RC,
                            unsigned NumElts) const;

  struct RegClassDescription {
    unsigned NumElements;
    unsigned ScalarBitSize;
  };

  // Maps TargetRegisterClass -> RegClassDescription
  DenseMap<const TargetRegisterClass *, std::unique_ptr<RegClassDescription>>
      RegClassMap;
  // Maps <NumElts, BitSize> -> TargetRegisterClass (vector reg class)
  DenseMap<std::pair<unsigned, unsigned>, const TargetRegisterClass *>
      VecRegClassMap;

  // map SubReg info to Swizzle
  struct SwizzleDesc {
    PISA::Swizzle Swizzle;
    const char *SwizzleName;
  };
  static DenseMap<unsigned, SwizzleDesc> SwizzleMap;

  BitVector SpecialRegs;
};
} // namespace llvm

#endif // LLVM_LIB_TARGET_PISA_PISAREGISTERINFO_H
