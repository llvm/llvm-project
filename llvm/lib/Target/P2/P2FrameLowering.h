//===-- P2FrameLowering.h - Define frame lowering for P2 ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIB_TARGET_P2_P2FRAMELOWERING_H
#define LLVM_LIB_TARGET_P2_P2FRAMELOWERING_H

#include "P2.h"

#include "llvm/CodeGen/TargetFrameLowering.h"

namespace llvm {

    class P2TargetMachine;

    class P2FrameLowering : public TargetFrameLowering {

        const P2TargetMachine &tm;

    public:
        explicit P2FrameLowering(const P2TargetMachine &TM)
            : TargetFrameLowering(StackGrowsUp, Align(1), 0), tm(TM) {
        }

        bool hasFP(const MachineFunction &MF) const override;

        /// emitProlog/emitEpilog - These methods insert prolog and epilog code into
        /// the function.
        void emitPrologue(MachineFunction &MF, MachineBasicBlock &MBB) const override;
        void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const override;

        void determineCalleeSaves(MachineFunction &MF, BitVector &SavedRegs, RegScavenger *RS=nullptr) const override;

        bool spillCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                        ArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const override;

        bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                                        MutableArrayRef<CalleeSavedInfo> CSI, const TargetRegisterInfo *TRI) const override;

        MachineBasicBlock::iterator eliminateCallFramePseudoInstr(MachineFunction &MF,
                                        MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator I) const override;

    };

} // End llvm namespace

#endif