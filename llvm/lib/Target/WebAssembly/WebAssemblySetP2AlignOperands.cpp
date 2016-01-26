//=- WebAssemblySetP2AlignOperands.cpp - Set alignments on loads and stores -=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file sets the p2align operands on load and store instructions.
///
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-set-p2align-operands"

namespace {
class WebAssemblySetP2AlignOperands final : public MachineFunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid
  WebAssemblySetP2AlignOperands() : MachineFunctionPass(ID) {}

  const char *getPassName() const override {
    return "WebAssembly Set p2align Operands";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    AU.addPreserved<MachineBlockFrequencyInfo>();
    AU.addPreservedID(MachineDominatorsID);
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};
} // end anonymous namespace

char WebAssemblySetP2AlignOperands::ID = 0;
FunctionPass *llvm::createWebAssemblySetP2AlignOperands() {
  return new WebAssemblySetP2AlignOperands();
}

bool WebAssemblySetP2AlignOperands::runOnMachineFunction(MachineFunction &MF) {
  DEBUG({
    dbgs() << "********** Set p2align Operands **********\n"
           << "********** Function: " << MF.getName() << '\n';
  });

  bool Changed = false;

  for (auto &MBB : MF) {
    for (auto &MI : MBB) {
      switch (MI.getOpcode()) {
      case WebAssembly::LOAD_I32:
      case WebAssembly::LOAD_I64:
      case WebAssembly::LOAD_F32:
      case WebAssembly::LOAD_F64:
      case WebAssembly::LOAD8_S_I32:
      case WebAssembly::LOAD8_U_I32:
      case WebAssembly::LOAD16_S_I32:
      case WebAssembly::LOAD16_U_I32:
      case WebAssembly::LOAD8_S_I64:
      case WebAssembly::LOAD8_U_I64:
      case WebAssembly::LOAD16_S_I64:
      case WebAssembly::LOAD16_U_I64:
      case WebAssembly::LOAD32_S_I64:
      case WebAssembly::LOAD32_U_I64:
      case WebAssembly::STORE_I32:
      case WebAssembly::STORE_I64:
      case WebAssembly::STORE_F32:
      case WebAssembly::STORE_F64:
      case WebAssembly::STORE8_I32:
      case WebAssembly::STORE16_I32:
      case WebAssembly::STORE8_I64:
      case WebAssembly::STORE16_I64:
      case WebAssembly::STORE32_I64:
        assert(MI.getOperand(3).getImm() == 0 &&
               "ISel should set p2align operands to 0");
        assert(MI.hasOneMemOperand() &&
               "Load and store instructions have exactly one mem operand");
        assert((*MI.memoperands_begin())->getSize() ==
                   (UINT64_C(1)
                    << WebAssembly::GetDefaultP2Align(MI.getOpcode())) &&
               "Default p2align value should be natural");
        assert(MI.getDesc().OpInfo[3].OperandType ==
                   WebAssembly::OPERAND_P2ALIGN &&
               "Load and store instructions should have a p2align operand");
        MI.getOperand(3).setImm(
            Log2_64((*MI.memoperands_begin())->getAlignment()));
        break;
      default:
        break;
      }
    }
  }

  return Changed;
}
