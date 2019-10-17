//===- DPUMacroFusion.h - DPU Macro Fusion --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "DPUMacroFusion.h"
#include "DPUSubtarget.h"
#include "llvm/CodeGen/MacroFusion.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dpu-mf"

#define GET_INSTRINFO_ENUM

#include "DPUGenInstrInfo.inc"

using namespace llvm;

static bool shouldScheduleAdjacent(const TargetInstrInfo &TII,
                                   const TargetSubtargetInfo &TSI,
                                   const MachineInstr *FirstMI,
                                   const MachineInstr &SecondMI) {
  // We are mainly interested in merging a simple operation with a simple
  // conditional/unconditional branch
  LLVM_DEBUG({
    dbgs() << "DPU/Merge: checking macro fusion:\n\t";
    if (!FirstMI)
      dbgs() << "<NONE>";
    else
      FirstMI->dump();
    dbgs() << "\n\t";
    SecondMI.dump();
    dbgs() << "\n";
  });

  if (!FirstMI) {
    // The second MI could be fused with any of the instructions of the
    // preceding block. Return true to trigger a test against any other
    // instruction.
    return true;
  }

  unsigned firstOpc = FirstMI->getOpcode();
  unsigned secondOpc = SecondMI.getOpcode();

  switch (secondOpc) {
  default:
    // todo probably more opportunities (Conditional branches...)
    return false;
  case DPU::JUMPi:
  case DPU::TmpJcci:
    break;
  case DPU::Jcci:
    if (!(FirstMI->getOperand(0).isReg() && SecondMI.getOperand(1).isReg() &&
          (FirstMI->getOperand(0).getReg() ==
           SecondMI.getOperand(1).getReg()))) {
      return false;
    }
    break;
  }

  switch (firstOpc) {
  default:
    // todo probably more opportunities (Operations with specific immediate
    // operands, call...)
    LLVM_DEBUG(dbgs() << "DPU/Merge: the two instructions cannot be fused\n");
    return false;
  case DPU::ADDrri:
  case DPU::ADDrrr:
  case DPU::SUBrir:
  case DPU::SUBrrr:
  case DPU::ORrri:
  case DPU::ORrrr:
  case DPU::ANDrrr:
  case DPU::ANDrri:
  case DPU::XORrri:
  case DPU::XORrrr:
  case DPU::NOTrr:
  case DPU::LSLrri:
  case DPU::LSLrrr:
  case DPU::LSRrrr:
  case DPU::LSRrri:
  case DPU::ASRrrr:
  case DPU::ASRrri:
  case DPU::ROLrrr:
  case DPU::ROLrri:
  case DPU::RORrrr:
  case DPU::RORrri:
  case DPU::CLZrr:
  case DPU::CAOrr:
  case DPU::MUL_UL_ULrrr:
  case DPU::MUL_SL_ULrrr:
  case DPU::MUL_SL_SLrrr:
  case DPU::MOVErr:
  case DPU::MOVEri:
    LLVM_DEBUG(dbgs() << "DPU/Merge: the two instructions can be fused\n");
    break;
  }

  return true;
}

namespace llvm {

std::unique_ptr<ScheduleDAGMutation> createDPUMacroFusionDAGMutation() {
  return createMacroFusionDAGMutation(shouldScheduleAdjacent);
}

} // end namespace llvm
