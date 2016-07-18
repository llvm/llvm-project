//===- OptimizationDiagnosticInfo.h - Optimization Diagnostic ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Optimization diagnostic interfaces.  It's packaged as an analysis pass so
// that by using this service passes become dependent on BFI as well.  BFI is
// used to compute the "hotness" of the diagnostic message.
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_OPTIMIZATIONDIAGNOSTICINFO_H
#define LLVM_IR_OPTIMIZATIONDIAGNOSTICINFO_H

#include "llvm/ADT/Optional.h"
#include "llvm/Pass.h"

namespace llvm {
class BlockFrequencyInfo;
class DebugLoc;
class Function;
class LLVMContext;
class Loop;
class Pass;
class Twine;
class Value;

class OptimizationRemarkEmitter : public FunctionPass {
public:
  OptimizationRemarkEmitter();

  /// Emit an optimization-missed message.
  ///
  /// \p PassName is the name of the pass emitting the message. If
  /// -Rpass-missed= is given and the name matches the regular expression in
  /// -Rpass, then the remark will be emitted. \p Fn is the function triggering
  /// the remark, \p DLoc is the debug location where the diagnostic is
  /// generated. \p V is the IR Value that identifies the code region. \p Msg is
  /// the message string to use.
  void emitOptimizationRemarkMissed(const char *PassName, const DebugLoc &DLoc,
                                    Value *V, const Twine &Msg);

  /// \brief Same as above but derives the IR Value for the code region and the
  /// debug location from the Loop parameter \p L.
  void emitOptimizationRemarkMissed(const char *PassName, Loop *L,
                                    const Twine &Msg);

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  static char ID;

private:
  Function *F;

  BlockFrequencyInfo *BFI;

  Optional<uint64_t> computeHotness(Value *V);
};
}

#endif // LLVM_IR_OPTIMIZATIONDIAGNOSTICINFO_H
