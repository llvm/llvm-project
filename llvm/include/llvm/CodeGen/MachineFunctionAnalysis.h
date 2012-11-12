//===-- MachineFunctionAnalysis.h - Owner of MachineFunctions ----*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the MachineFunctionAnalysis class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_FUNCTION_ANALYSIS_H
#define LLVM_CODEGEN_MACHINE_FUNCTION_ANALYSIS_H

#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {

class MachineFunction;

/// MachineFunctionAnalysis - This class is a Pass that manages a
/// MachineFunction object.
struct MachineFunctionAnalysis : public FunctionPass {
private:
  const TargetMachine &TM;
  MachineFunction *MF;
  unsigned NextFnNum;
public:
  static char ID;
  explicit MachineFunctionAnalysis(const TargetMachine &tm);
  ~MachineFunctionAnalysis();

  MachineFunction &getMF() const { return *MF; }
  
  virtual const char* getPassName() const {
    return "Machine Function Analysis";
  }

private:
  virtual bool doInitialization(Module &M);
  virtual bool runOnFunction(Function &F);
  virtual void releaseMemory();
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};

} // End llvm namespace

#endif
