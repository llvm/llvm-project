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
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

namespace llvm {
class DebugLoc;
class Function;
class LLVMContext;
class Loop;
class Pass;
class Twine;
class Value;

/// The optimization diagnostic interface.
///
/// It allows reporting when optimizations are performed and when they are not
/// along with the reasons for it.  Hotness information of the corresponding
/// code region can be included in the remark if DiagnosticHotnessRequested is
/// enabled in the LLVM context.
class OptimizationRemarkEmitter {
public:
  OptimizationRemarkEmitter(Function *F, BlockFrequencyInfo *BFI)
      : F(F), BFI(BFI) {}

  /// \brief This variant can be used to generate ORE on demand (without the
  /// analysis pass).
  ///
  /// Note that this ctor has a very different cost depending on whether
  /// F->getContext().getDiagnosticHotnessRequested() is on or not.  If it's off
  /// the operation is free.
  ///
  /// Whereas if DiagnosticHotnessRequested is on, it is fairly expensive
  /// operation since BFI and all its required analyses are computed.  This is
  /// for example useful for CGSCC passes that can't use function analyses
  /// passes in the old PM.
  OptimizationRemarkEmitter(Function *F);

  OptimizationRemarkEmitter(OptimizationRemarkEmitter &&Arg)
      : F(Arg.F), BFI(Arg.BFI) {}

  OptimizationRemarkEmitter &operator=(OptimizationRemarkEmitter &&RHS) {
    F = RHS.F;
    BFI = RHS.BFI;
    return *this;
  }

  /// Emit an optimization-applied message.
  ///
  /// \p PassName is the name of the pass emitting the message. If -Rpass= is
  /// given and \p PassName matches the regular expression in -Rpass, then the
  /// remark will be emitted. \p Fn is the function triggering the remark, \p
  /// DLoc is the debug location where the diagnostic is generated. \p V is the
  /// IR Value that identifies the code region. \p Msg is the message string to
  /// use.
  void emitOptimizationRemark(const char *PassName, const DebugLoc &DLoc,
                              const Value *V, const Twine &Msg);

  /// \brief Same as above but derives the IR Value for the code region and the
  /// debug location from the Loop parameter \p L.
  void emitOptimizationRemark(const char *PassName, Loop *L, const Twine &Msg);

  /// \brief Same as above but derives the debug location and the code region
  /// from the debug location and the basic block of \p Inst, respectively.
  void emitOptimizationRemark(const char *PassName, Instruction *Inst,
                              const Twine &Msg) {
    emitOptimizationRemark(PassName, Inst->getDebugLoc(), Inst->getParent(),
                           Msg);
  }

  /// Emit an optimization-missed message.
  ///
  /// \p PassName is the name of the pass emitting the message. If
  /// -Rpass-missed= is given and the name matches the regular expression in
  /// -Rpass, then the remark will be emitted.  \p DLoc is the debug location
  /// where the diagnostic is generated. \p V is the IR Value that identifies
  /// the code region. \p Msg is the message string to use.  If \p IsVerbose is
  /// true, the message is considered verbose and will only be emitted when
  /// verbose output is turned on.
  void emitOptimizationRemarkMissed(const char *PassName, const DebugLoc &DLoc,
                                    const Value *V, const Twine &Msg,
                                    bool IsVerbose = false);

  /// \brief Same as above but derives the IR Value for the code region and the
  /// debug location from the Loop parameter \p L.
  void emitOptimizationRemarkMissed(const char *PassName, Loop *L,
                                    const Twine &Msg, bool IsVerbose = false);

  /// \brief Same as above but derives the debug location and the code region
  /// from the debug location and the basic block of \p Inst, respectively.
  void emitOptimizationRemarkMissed(const char *PassName, Instruction *Inst,
                                    const Twine &Msg, bool IsVerbose = false) {
    emitOptimizationRemarkMissed(PassName, Inst->getDebugLoc(),
                                 Inst->getParent(), Msg, IsVerbose);
  }

  /// Emit an optimization analysis remark message.
  ///
  /// \p PassName is the name of the pass emitting the message. If
  /// -Rpass-analysis= is given and \p PassName matches the regular expression
  /// in -Rpass, then the remark will be emitted. \p DLoc is the debug location
  /// where the diagnostic is generated. \p V is the IR Value that identifies
  /// the code region. \p Msg is the message string to use. If \p IsVerbose is
  /// true, the message is considered verbose and will only be emitted when
  /// verbose output is turned on.
  void emitOptimizationRemarkAnalysis(const char *PassName,
                                      const DebugLoc &DLoc, const Value *V,
                                      const Twine &Msg, bool IsVerbose = false);

  /// \brief Same as above but derives the IR Value for the code region and the
  /// debug location from the Loop parameter \p L.
  void emitOptimizationRemarkAnalysis(const char *PassName, Loop *L,
                                      const Twine &Msg, bool IsVerbose = false);

  /// \brief Same as above but derives the debug location and the code region
  /// from the debug location and the basic block of \p Inst, respectively.
  void emitOptimizationRemarkAnalysis(const char *PassName, Instruction *Inst,
                                      const Twine &Msg,
                                      bool IsVerbose = false) {
    emitOptimizationRemarkAnalysis(PassName, Inst->getDebugLoc(),
                                   Inst->getParent(), Msg, IsVerbose);
  }

  /// \brief This variant allows specifying what should be emitted for missed
  /// and analysis remarks in one call.
  ///
  /// \p PassName is the name of the pass emitting the message. If
  /// -Rpass-missed= is given and \p PassName matches the regular expression, \p
  /// MsgForMissedRemark is emitted.
  ///
  /// If -Rpass-analysis= is given and \p PassName matches the regular
  /// expression, \p MsgForAnalysisRemark is emitted.
  ///
  /// The debug location and the code region is derived from \p Inst. If \p
  /// IsVerbose is true, the message is considered verbose and will only be
  /// emitted when verbose output is turned on.
  void emitOptimizationRemarkMissedAndAnalysis(
      const char *PassName, Instruction *Inst, const Twine &MsgForMissedRemark,
      const Twine &MsgForAnalysisRemark, bool IsVerbose = false) {
    emitOptimizationRemarkAnalysis(PassName, Inst, MsgForAnalysisRemark,
                                   IsVerbose);
    emitOptimizationRemarkMissed(PassName, Inst, MsgForMissedRemark, IsVerbose);
  }

  /// \brief Emit an optimization analysis remark related to floating-point
  /// non-commutativity.
  ///
  /// \p PassName is the name of the pass emitting the message. If
  /// -Rpass-analysis= is given and \p PassName matches the regular expression
  /// in -Rpass, then the remark will be emitted. \p Fn is the function
  /// triggering the remark, \p DLoc is the debug location where the diagnostic
  /// is generated.\p V is the IR Value that identifies the code region.  \p Msg
  /// is the message string to use.
  void emitOptimizationRemarkAnalysisFPCommute(const char *PassName,
                                               const DebugLoc &DLoc,
                                               const Value *V,
                                               const Twine &Msg);

  /// \brief Emit an optimization analysis remark related to pointer aliasing.
  ///
  /// \p PassName is the name of the pass emitting the message. If
  /// -Rpass-analysis= is given and \p PassName matches the regular expression
  /// in -Rpass, then the remark will be emitted. \p Fn is the function
  /// triggering the remark, \p DLoc is the debug location where the diagnostic
  /// is generated.\p V is the IR Value that identifies the code region.  \p Msg
  /// is the message string to use.
  void emitOptimizationRemarkAnalysisAliasing(const char *PassName,
                                              const DebugLoc &DLoc,
                                              const Value *V, const Twine &Msg);

  /// \brief Same as above but derives the IR Value for the code region and the
  /// debug location from the Loop parameter \p L.
  void emitOptimizationRemarkAnalysisAliasing(const char *PassName, Loop *L,
                                              const Twine &Msg);

private:
  Function *F;

  BlockFrequencyInfo *BFI;

  /// If we generate BFI on demand, we need to free it when ORE is freed.
  std::unique_ptr<BlockFrequencyInfo> OwnedBFI;

  Optional<uint64_t> computeHotness(const Value *V);

  /// \brief Only allow verbose messages if we know we're filtering by hotness
  /// (BFI is only set in this case).
  bool shouldEmitVerbose() { return BFI != nullptr; }

  OptimizationRemarkEmitter(const OptimizationRemarkEmitter &) = delete;
  void operator=(const OptimizationRemarkEmitter &) = delete;
};

/// OptimizationRemarkEmitter legacy analysis pass
///
/// Note that this pass shouldn't generally be marked as preserved by other
/// passes.  It's holding onto BFI, so if the pass does not preserve BFI, BFI
/// could be freed.
class OptimizationRemarkEmitterWrapperPass : public FunctionPass {
  std::unique_ptr<OptimizationRemarkEmitter> ORE;

public:
  OptimizationRemarkEmitterWrapperPass();

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  OptimizationRemarkEmitter &getORE() {
    assert(ORE && "pass not run yet");
    return *ORE;
  }

  static char ID;
};

class OptimizationRemarkEmitterAnalysis
    : public AnalysisInfoMixin<OptimizationRemarkEmitterAnalysis> {
  friend AnalysisInfoMixin<OptimizationRemarkEmitterAnalysis>;
  static char PassID;

public:
  /// \brief Provide the result typedef for this analysis pass.
  typedef OptimizationRemarkEmitter Result;

  /// \brief Run the analysis pass over a function and produce BFI.
  Result run(Function &F, FunctionAnalysisManager &AM);
};
}
#endif // LLVM_IR_OPTIMIZATIONDIAGNOSTICINFO_H
