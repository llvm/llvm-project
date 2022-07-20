//
// Created by tanmay on 6/12/22.
//

#include "llvm/Transforms/ErrorAnalysis/ErrorAnalysis.h"
#include "llvm/Transforms/ErrorAnalysis/AtomicCondition/ACInstrumentation.h"
#include "llvm/IR/Function.h"

using namespace llvm;
using namespace atomiccondition;

PreservedAnalyses ErrorAnalysisPass::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  // Discarding function declarations
  if(F.isDeclaration())
    return PreservedAnalyses::all();

  if (ACInstrumentation::isUnwantedFunction(&F))
    return PreservedAnalyses::all();

  errs() << "---------------------------------------------------" << "\n";
  errs() << "Current Function: " << F.getName() << "\n";
  errs() << "---------------------------------------------------" << "\n";

  Function *FunctionPointer = &F;
  ACInstrumentation *InstrumentationObject = new ACInstrumentation(FunctionPointer);
  long int NumInstrumentedInstructionsInF = 0;

  if(F.getName().compare("main")==0)
    InstrumentationObject->instrumentMainFunction(&F);

  for(Function::iterator CurrentBB = FunctionPointer->begin();
       CurrentBB != FunctionPointer->end();
       ++CurrentBB) {

    BasicBlock *BB = &*CurrentBB;

    errs() << "Current Basic Block: " << BB->getName() << "\n";

    long int NumInstrumentedInstructionsInBB = 0;
    InstrumentationObject->instrumentBasicBlock(BB, &NumInstrumentedInstructionsInBB);
    NumInstrumentedInstructionsInF += NumInstrumentedInstructionsInBB;

    errs() << std::to_string(NumInstrumentedInstructionsInBB) <<
        " instructions instrumented in " << BB->getName() << "\n";
  }

  errs() << std::to_string(NumInstrumentedInstructionsInF) <<
      " instructions instrumented in " << F.getName() << "\n";
  errs() << "---------------------------------------------------" << "\n\n";

  delete InstrumentationObject;

  return PreservedAnalyses::all();
}