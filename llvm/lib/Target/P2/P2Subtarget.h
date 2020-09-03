//===-- P2Subtarget.h - Define Subtarget for the P2 ----*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the P2 specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_P2_P2SUBTARGET_H
#define LLVM_LIB_TARGET_P2_P2SUBTARGET_H

#include "P2FrameLowering.h"
#include "P2ISelLowering.h"
#include "P2InstrInfo.h"
#include "P2RegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGTargetInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/DataLayout.h"
#include <string>

#define GET_SUBTARGETINFO_HEADER
#include "P2GenSubtargetInfo.inc"

namespace llvm {
    class StringRef;
    class TargetMachine;

    class P2Subtarget : public P2GenSubtargetInfo {
        virtual void anchor();

        P2FrameLowering FrameLowering;
        P2InstrInfo InstrInfo;
        P2TargetLowering TLInfo;
        SelectionDAGTargetInfo TSInfo;

    public:
        // implementation generated in P2GenSubtargetInfo.inc
        void ParseSubtargetFeatures(StringRef CPU, StringRef TuneCPU, StringRef FS);

        /// This constructor initializes the data members to match that
        /// of the specified triple.
        ///
        P2Subtarget(const Triple &TT, const std::string &CPU, const std::string &FS, const P2TargetMachine &TM);

        const TargetFrameLowering *getFrameLowering() const override { return &FrameLowering; }
        const P2InstrInfo *getInstrInfo() const override { return &InstrInfo; }
        const TargetRegisterInfo *getRegisterInfo() const override { return &InstrInfo.getRegisterInfo(); }
        const P2TargetLowering *getTargetLowering() const override { return &TLInfo; }
        const SelectionDAGTargetInfo *getSelectionDAGInfo() const override { return &TSInfo; }

    };

} // End llvm namespace

#endif  // LLVM_TARGET_P2_SUBTARGET_H