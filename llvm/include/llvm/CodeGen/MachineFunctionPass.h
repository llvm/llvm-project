//===-- MachineFunctionPass.h - Pass for MachineFunctions --------*-C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachineFunctionPass class.  MachineFunctionPass's are
// just FunctionPass's, except they operate on machine code as part of a code
// generator.  Because they operate on machine code, not the LLVM
// representation, MachineFunctionPass's are not allowed to modify the LLVM
// representation.  Due to this limitation, the MachineFunctionPass class takes
// care of declaring that no LLVM passes are invalidated.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINE_FUNCTION_PASS_H
#define LLVM_CODEGEN_MACHINE_FUNCTION_PASS_H

#include "llvm/Pass.h"

namespace llvm {

class MachineFunction;

/// MachineFunctionPass - This class adapts the FunctionPass interface to
/// allow convenient creation of passes that operate on the MachineFunction
/// representation. Instead of overriding runOnFunction, subclasses
/// override runOnMachineFunction.
class MachineFunctionPass : public FunctionPass {
protected:
  explicit MachineFunctionPass(char &ID) : FunctionPass(ID) {}

  /// runOnMachineFunction - This method must be overloaded to perform the
  /// desired machine code transformation or analysis.
  ///
  virtual bool runOnMachineFunction(MachineFunction &MF) = 0;

  /// getAnalysisUsage - Subclasses that override getAnalysisUsage
  /// must call this.
  ///
  /// For MachineFunctionPasses, calling AU.preservesCFG() indicates that
  /// the pass does not modify the MachineBasicBlock CFG.
  ///
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

private:
  /// createPrinterPass - Get a machine function printer pass.
  virtual Pass *createPrinterPass(raw_ostream &O,
                                  const std::string &Banner) const;

  virtual bool runOnFunction(Function &F);
};

} // End llvm namespace

#endif
