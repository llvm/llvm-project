#include "llvm/Transforms/Utils/DebugRecordCountPass.h"

#include "llvm/IR/DebugProgramInstruction.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

PreservedAnalyses DebugRecordCountPass::run(Function &F, FunctionAnalysisManager &) {
  int dbgValue = 0;
  int dbgDeclare = 0;
  int dbgAssign = 0;

  for (auto &BB : F) {
    for (auto &I : BB) {
        for(auto &D : I.getDbgRecordRange()) {
            if(auto *db = dyn_cast<DbgVariableRecord>(&D)) {
                if(db->isDbgValue()) dbgValue++;
                else if(db->isDbgDeclare()) dbgDeclare++;
                else if(db->isDbgAssign()) dbgAssign++;
            }
        }
    }
  }

  outs() << "Function: " << F.getName() << "\n";
  outs() << "\t#dbg_values : " << dbgValue << "\n";
  outs() << "\t#dbg_declare: " << dbgDeclare << "\n";
  outs() << "\t#dbg_assign : " << dbgAssign << "\n";

  return PreservedAnalyses::all();
}