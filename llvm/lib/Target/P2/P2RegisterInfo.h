//===-- P2RegisterInfo.h - P2 Register Information Impl -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the P2 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2REGISTERINFO_H
#define LLVM_LIB_TARGET_P2_P2REGISTERINFO_H

#include "P2.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "P2GenRegisterInfo.inc"

namespace llvm {
    class TargetInstrInfo;
    class Type;

    class P2RegisterInfo : public P2GenRegisterInfo {

    public:
        P2RegisterInfo();

        const MCPhysReg *getCalleeSavedRegs(const MachineFunction *MF) const override;
        const uint32_t *getCallPreservedMask(const MachineFunction &MF, CallingConv::ID) const override;

        BitVector getReservedRegs(const MachineFunction &MF) const override;

        bool requiresRegisterScavenging(const MachineFunction &MF) const override {
            return true;
        }

        /// Stack Frame Processing Methods
        void eliminateFrameIndex(MachineBasicBlock::iterator II, int SPAdj, unsigned FIOperandNum, RegScavenger *RS = nullptr) const override;

        /// Debug information queries.
        Register getFrameRegister(const MachineFunction &MF) const override;
    };

} // end namespace llvm

#endif