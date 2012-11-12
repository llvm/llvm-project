//===- HexagonRemoveExtendArgs.cpp - Remove unnecessary argument sign extends //
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Pass that removes sign extends for function parameters. These parameters
// are already sign extended by the caller per Hexagon's ABI
//
//===----------------------------------------------------------------------===//

#include "HexagonTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/CodeGen/MachineFunctionAnalysis.h"
#include "llvm/Transforms/Scalar.h"

using namespace llvm;
namespace {
  struct HexagonRemoveExtendArgs : public FunctionPass {
  public:
    static char ID;
    HexagonRemoveExtendArgs() : FunctionPass(ID) {}
    virtual bool runOnFunction(Function &F);

    const char *getPassName() const {
      return "Remove sign extends";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<MachineFunctionAnalysis>();
      AU.addPreserved<MachineFunctionAnalysis>();
      FunctionPass::getAnalysisUsage(AU);
    }
  };
}

char HexagonRemoveExtendArgs::ID = 0;
RegisterPass<HexagonRemoveExtendArgs> X("reargs",
                                        "Remove Sign and Zero Extends for Args"
                                        );



bool HexagonRemoveExtendArgs::runOnFunction(Function &F) {
  unsigned Idx = 1;
  for (Function::arg_iterator AI = F.arg_begin(), AE = F.arg_end(); AI != AE;
       ++AI, ++Idx) {
    if (F.getParamAttributes(Idx).hasAttribute(Attributes::SExt)) {
      Argument* Arg = AI;
      if (!isa<PointerType>(Arg->getType())) {
        for (Instruction::use_iterator UI = Arg->use_begin();
             UI != Arg->use_end();) {
          if (isa<SExtInst>(*UI)) {
            Instruction* Use = cast<Instruction>(*UI);
            SExtInst* SI = new SExtInst(Arg, Use->getType());
            assert (EVT::getEVT(SI->getType()) ==
                    (EVT::getEVT(Use->getType())));
            ++UI;
            Use->replaceAllUsesWith(SI);
            Instruction* First = F.getEntryBlock().begin();
            SI->insertBefore(First);
            Use->eraseFromParent();
          } else {
            ++UI;
          }
        }
      }
    }
  }
  return true;
}



FunctionPass *llvm::createHexagonRemoveExtendOps(HexagonTargetMachine &TM) {
  return new HexagonRemoveExtendArgs();
}
