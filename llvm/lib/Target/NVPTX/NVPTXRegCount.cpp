//===-- NVPTXISelDAGToDAG.cpp - A dag to dag inst selector for NVPTX ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the NVPTX target.
//
//===----------------------------------------------------------------------===//

#include "NVPTXISelDAGToDAG.h"
#include "NVPTX.h"
#include "NVPTXUtilities.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/NVVMIntrinsicUtils.h"
#include "llvm/Support/AtomicOrdering.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>
using namespace llvm;

namespace {
class NVPTXRegCountPass : public MachineFunctionPass {
    public:
      static char ID;
      NVPTXRegCountPass() : MachineFunctionPass(ID) {}
    
      bool runOnMachineFunction(MachineFunction &MF) override {
        unsigned maxRegs = 0;
        for (const MachineBasicBlock &MBB : MF) {
          unsigned liveRegs = 0;
          for (const MachineInstr &MI : MBB) {
            // Count unique virtual and physical registers
            for (const MachineOperand &MO : MI.operands()) {
              if (MO.isReg() && MO.getReg())
                liveRegs++;
            }
          }
          maxRegs = std::max(maxRegs, liveRegs);
        }
        errs() << "Function " << MF.getName() << " uses maximum of " 
               << maxRegs << " registers\n";
        return false;
      }
    };
} // namespace

char NVPTXRegCountPass::ID = 0;
// INITIALIZE_PASS(NVPTXRegCountPass, "nvptx-count-reg",
//     "NVPTX count reg", false, false)

    FunctionPass *llvm::createNVPTXRegCountPass() {
      return new NVPTXRegCountPass();
    }