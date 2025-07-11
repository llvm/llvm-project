//===------- VRegMaskPair.h ----------------------------------------*- C++-
//*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_VREGMASKPAIR_H
#define LLVM_LIB_TARGET_VREGMASKPAIR_H

#include "llvm/CodeGen/Register.h"
#include "llvm/MC/LaneBitmask.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/Compiler.h"
#include <cassert>

class VRegMaskPair {
    
      Register VReg;
      LaneBitmask LaneMask;

    public:
      VRegMaskPair(Register VReg, LaneBitmask LaneMask)
          : VReg(VReg), LaneMask(LaneMask) {}

      VRegMaskPair()
          : VReg(AMDGPU::NoRegister), LaneMask(LaneBitmask::getNone()) {}
      VRegMaskPair(const VRegMaskPair &Other) = default;
      VRegMaskPair(VRegMaskPair &&Other) = default;
      VRegMaskPair &operator=(const VRegMaskPair &Other) = default;
      VRegMaskPair &operator=(VRegMaskPair &&Other) = default;

      VRegMaskPair(const MachineOperand MO, const SIRegisterInfo *TRI,
                   const MachineRegisterInfo *MRI) {
        assert(MO.isReg() && "Not a register operand!");
        Register R = MO.getReg();
        const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, R);
        assert(R.isVirtual() && "Not a virtual register!");
        VReg = R;
        LaneMask = getFullMaskForRC(*RC, TRI);
        unsigned subRegIndex = MO.getSubReg();
        if (subRegIndex) {
          LaneMask = TRI->getSubRegIndexLaneMask(subRegIndex);
        }
      }

      const Register getVReg() const { return VReg; }
      const LaneBitmask getLaneMask() const { return LaneMask; }

      unsigned getSubReg(const MachineRegisterInfo *MRI,
                         const SIRegisterInfo *TRI) const {
        const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
        LaneBitmask Mask = getFullMaskForRC(*RC, TRI);
        if (LaneMask != Mask)
          return getSubRegIndexForLaneMask(LaneMask, TRI);
        return AMDGPU::NoRegister;
      }

      const TargetRegisterClass *getRegClass(const MachineRegisterInfo *MRI,
                                             const SIRegisterInfo *TRI) const {

        const TargetRegisterClass *RC = TRI->getRegClassForReg(*MRI, VReg);
        LaneBitmask Mask = getFullMaskForRC(*RC, TRI);
        if (LaneMask != Mask) {
          unsigned SubRegIdx = getSubRegIndexForLaneMask(LaneMask, TRI);
          // RC = TRI->getSubRegisterClass(RC, SubRegIdx);
          return TRI->getSubRegisterClass(RC, SubRegIdx);
        }

        return RC;
      }

      unsigned getSizeInRegs(const MachineRegisterInfo *MRI,
                             const SIRegisterInfo *TRI) const {
        const TargetRegisterClass *RC = getRegClass(MRI, TRI);
        return TRI->getRegClassWeight(RC).RegWeight;
      }

      bool operator==(const VRegMaskPair &other) const {
        return VReg == other.VReg && LaneMask == other.LaneMask;
      }
    };
    
    namespace llvm {
    template <> struct DenseMapInfo<VRegMaskPair> {
      static inline VRegMaskPair getEmptyKey() {
        return {Register(DenseMapInfo<unsigned>::getEmptyKey()),
                LaneBitmask(0xFFFFFFFFFFFFFFFFULL)};
      }
    
      static inline VRegMaskPair getTombstoneKey() {
        return {Register(DenseMapInfo<unsigned>::getTombstoneKey()),
                LaneBitmask(0xFFFFFFFFFFFFFFFEULL)};
      }

      static unsigned getHashValue(const VRegMaskPair &P) {
        return DenseMapInfo<unsigned>::getHashValue(P.getVReg().id()) ^
               DenseMapInfo<uint64_t>::getHashValue(
                   P.getLaneMask().getAsInteger());
      }

      static bool isEqual(const VRegMaskPair &LHS, const VRegMaskPair &RHS) {
        return DenseMapInfo<unsigned>::isEqual(LHS.getVReg().id(),
                                               RHS.getVReg().id()) &&
               DenseMapInfo<uint64_t>::isEqual(
                   LHS.getLaneMask().getAsInteger(),
                   RHS.getLaneMask().getAsInteger());
      }
    };
    } // namespace llvm
#endif // LLVM_LIB_TARGET_VREGMASKPAIR_H