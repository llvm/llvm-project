//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

#include "llvm/Pass.h"
#include "llvm/Target/TargetMachine.h"
#include <string>

namespace llvm {

  class FunctionPass;
  class MachineFunctionPass;
  class PassInfo;
  class TargetLowering;
  class TargetRegisterClass;
  class raw_ostream;
}

namespace llvm {

extern char &NoPassID; // Allow targets to choose not to run a pass.

class PassConfigImpl;

/// Target-Independent Code Generator Pass Configuration Options.
///
/// This is an ImmutablePass solely for the purpose of exposing CodeGen options
/// to the internals of other CodeGen passes.
class TargetPassConfig : public ImmutablePass {
public:
  /// Pseudo Pass IDs. These are defined within TargetPassConfig because they
  /// are unregistered pass IDs. They are only useful for use with
  /// TargetPassConfig APIs to identify multiple occurrences of the same pass.
  ///

  /// EarlyTailDuplicate - A clone of the TailDuplicate pass that runs early
  /// during codegen, on SSA form.
  static char EarlyTailDuplicateID;

  /// PostRAMachineLICM - A clone of the LICM pass that runs during late machine
  /// optimization after regalloc.
  static char PostRAMachineLICMID;

protected:
  TargetMachine *TM;
  PassManagerBase *PM;
  PassConfigImpl *Impl; // Internal data structures
  bool Initialized;     // Flagged after all passes are configured.

  // Target Pass Options
  // Targets provide a default setting, user flags override.
  //
  bool DisableVerify;

  /// Default setting for -enable-tail-merge on this target.
  bool EnableTailMerge;

public:
  TargetPassConfig(TargetMachine *tm, PassManagerBase &pm);
  // Dummy constructor.
  TargetPassConfig();

  virtual ~TargetPassConfig();

  static char ID;

  /// Get the right type of TargetMachine for this target.
  template<typename TMC> TMC &getTM() const {
    return *static_cast<TMC*>(TM);
  }

  const TargetLowering *getTargetLowering() const {
    return TM->getTargetLowering();
  }

  //
  void setInitialized() { Initialized = true; }

  CodeGenOpt::Level getOptLevel() const { return TM->getOptLevel(); }

  void setDisableVerify(bool Disable) { setOpt(DisableVerify, Disable); }

  bool getEnableTailMerge() const { return EnableTailMerge; }
  void setEnableTailMerge(bool Enable) { setOpt(EnableTailMerge, Enable); }

  /// Allow the target to override a specific pass without overriding the pass
  /// pipeline. When passes are added to the standard pipeline at the
  /// point where StadardID is expected, add TargetID in its place.
  void substitutePass(char &StandardID, char &TargetID);

  /// Allow the target to enable a specific standard pass by default.
  void enablePass(char &ID) { substitutePass(ID, ID); }

  /// Allow the target to disable a specific standard pass by default.
  void disablePass(char &ID) { substitutePass(ID, NoPassID); }

  /// Return the pass ssubtituted for StandardID by the target.
  /// If no substitution exists, return StandardID.
  AnalysisID getPassSubstitution(AnalysisID StandardID) const;

  /// Return true if the optimized regalloc pipeline is enabled.
  bool getOptimizeRegAlloc() const;

  /// Add common target configurable passes that perform LLVM IR to IR
  /// transforms following machine independent optimization.
  virtual void addIRPasses();

  /// Add common passes that perform LLVM IR to IR transforms in preparation for
  /// instruction selection.
  virtual void addISelPrepare();

  /// addInstSelector - This method should install an instruction selector pass,
  /// which converts from LLVM code to machine instructions.
  virtual bool addInstSelector() {
    return true;
  }

  /// Add the complete, standard set of LLVM CodeGen passes.
  /// Fully developed targets will not generally override this.
  virtual void addMachinePasses();

protected:
  // Helper to verify the analysis is really immutable.
  void setOpt(bool &Opt, bool Val);

  /// Methods with trivial inline returns are convenient points in the common
  /// codegen pass pipeline where targets may insert passes. Methods with
  /// out-of-line standard implementations are major CodeGen stages called by
  /// addMachinePasses. Some targets may override major stages when inserting
  /// passes is insufficient, but maintaining overriden stages is more work.
  ///

  /// addPreISelPasses - This method should add any "last minute" LLVM->LLVM
  /// passes (which are run just before instruction selector).
  virtual bool addPreISel() {
    return true;
  }

  /// addMachineSSAOptimization - Add standard passes that optimize machine
  /// instructions in SSA form.
  virtual void addMachineSSAOptimization();

  /// addPreRegAlloc - This method may be implemented by targets that want to
  /// run passes immediately before register allocation. This should return
  /// true if -print-machineinstrs should print after these passes.
  virtual bool addPreRegAlloc() {
    return false;
  }

  /// createTargetRegisterAllocator - Create the register allocator pass for
  /// this target at the current optimization level.
  virtual FunctionPass *createTargetRegisterAllocator(bool Optimized);

  /// addFastRegAlloc - Add the minimum set of target-independent passes that
  /// are required for fast register allocation.
  virtual void addFastRegAlloc(FunctionPass *RegAllocPass);

  /// addOptimizedRegAlloc - Add passes related to register allocation.
  /// LLVMTargetMachine provides standard regalloc passes for most targets.
  virtual void addOptimizedRegAlloc(FunctionPass *RegAllocPass);

  /// addFinalizeRegAlloc - This method may be implemented by targets that want
  /// to run passes within the regalloc pipeline, immediately after the register
  /// allocation pass itself. These passes run as soon as virtual regisiters
  /// have been rewritten to physical registers but before and other postRA
  /// optimization happens. Targets that have marked instructions for bundling
  /// must have finalized those bundles by the time these passes have run,
  /// because subsequent passes are not guaranteed to be bundle-aware.
  virtual bool addFinalizeRegAlloc() {
    return false;
  }

  /// addPostRegAlloc - This method may be implemented by targets that want to
  /// run passes after register allocation pass pipeline but before
  /// prolog-epilog insertion.  This should return true if -print-machineinstrs
  /// should print after these passes.
  virtual bool addPostRegAlloc() {
    return false;
  }

  /// Add passes that optimize machine instructions after register allocation.
  virtual void addMachineLateOptimization();

  /// addPreSched2 - This method may be implemented by targets that want to
  /// run passes after prolog-epilog insertion and before the second instruction
  /// scheduling pass.  This should return true if -print-machineinstrs should
  /// print after these passes.
  virtual bool addPreSched2() {
    return false;
  }

  /// Add standard basic block placement passes.
  virtual void addBlockPlacement();

  /// addPreEmitPass - This pass may be implemented by targets that want to run
  /// passes immediately before machine code is emitted.  This should return
  /// true if -print-machineinstrs should print out the code after the passes.
  virtual bool addPreEmitPass() {
    return false;
  }

  /// Utilities for targets to add passes to the pass manager.
  ///

  /// Add a CodeGen pass at this point in the pipeline after checking overrides.
  /// Return the pass that was added, or NoPassID.
  AnalysisID addPass(char &ID);

  /// addMachinePasses helper to create the target-selected or overriden
  /// regalloc pass.
  FunctionPass *createRegAllocPass(bool Optimized);

  /// printAndVerify - Add a pass to dump then verify the machine function, if
  /// those steps are enabled.
  ///
  void printAndVerify(const char *Banner) const;
};
} // namespace llvm

/// List of target independent CodeGen pass IDs.
namespace llvm {
  /// createUnreachableBlockEliminationPass - The LLVM code generator does not
  /// work well with unreachable basic blocks (what live ranges make sense for a
  /// block that cannot be reached?).  As such, a code generator should either
  /// not instruction select unreachable blocks, or run this pass as its
  /// last LLVM modifying pass to clean up blocks that are not reachable from
  /// the entry block.
  FunctionPass *createUnreachableBlockEliminationPass();

  /// MachineFunctionPrinter pass - This pass prints out the machine function to
  /// the given stream as a debugging tool.
  MachineFunctionPass *
  createMachineFunctionPrinterPass(raw_ostream &OS,
                                   const std::string &Banner ="");

  /// MachineLoopInfo - This pass is a loop analysis pass.
  extern char &MachineLoopInfoID;

  /// MachineLoopRanges - This pass is an on-demand loop coverage analysis.
  extern char &MachineLoopRangesID;

  /// MachineDominators - This pass is a machine dominators analysis pass.
  extern char &MachineDominatorsID;

  /// EdgeBundles analysis - Bundle machine CFG edges.
  extern char &EdgeBundlesID;

  /// LiveVariables pass - This pass computes the set of blocks in which each
  /// variable is life and sets machine operand kill flags.
  extern char &LiveVariablesID;

  /// PHIElimination - This pass eliminates machine instruction PHI nodes
  /// by inserting copy instructions.  This destroys SSA information, but is the
  /// desired input for some register allocators.  This pass is "required" by
  /// these register allocator like this: AU.addRequiredID(PHIEliminationID);
  extern char &PHIEliminationID;

  /// StrongPHIElimination - This pass eliminates machine instruction PHI
  /// nodes by inserting copy instructions.  This destroys SSA information, but
  /// is the desired input for some register allocators.  This pass is
  /// "required" by these register allocator like this:
  ///    AU.addRequiredID(PHIEliminationID);
  ///  This pass is still in development
  extern char &StrongPHIEliminationID;

  /// LiveStacks pass. An analysis keeping track of the liveness of stack slots.
  extern char &LiveStacksID;

  /// TwoAddressInstruction - This pass reduces two-address instructions to
  /// use two operands. This destroys SSA information but it is desired by
  /// register allocators.
  extern char &TwoAddressInstructionPassID;

  /// ProcessImpicitDefs pass - This pass removes IMPLICIT_DEFs.
  extern char &ProcessImplicitDefsID;

  /// RegisterCoalescer - This pass merges live ranges to eliminate copies.
  extern char &RegisterCoalescerID;

  /// MachineScheduler - This pass schedules machine instructions.
  extern char &MachineSchedulerID;

  /// SpillPlacement analysis. Suggest optimal placement of spill code between
  /// basic blocks.
  extern char &SpillPlacementID;

  /// UnreachableMachineBlockElimination - This pass removes unreachable
  /// machine basic blocks.
  extern char &UnreachableMachineBlockElimID;

  /// DeadMachineInstructionElim - This pass removes dead machine instructions.
  extern char &DeadMachineInstructionElimID;

  /// FastRegisterAllocation Pass - This pass register allocates as fast as
  /// possible. It is best suited for debug code where live ranges are short.
  ///
  FunctionPass *createFastRegisterAllocator();

  /// BasicRegisterAllocation Pass - This pass implements a degenerate global
  /// register allocator using the basic regalloc framework.
  ///
  FunctionPass *createBasicRegisterAllocator();

  /// Greedy register allocation pass - This pass implements a global register
  /// allocator for optimized builds.
  ///
  FunctionPass *createGreedyRegisterAllocator();

  /// PBQPRegisterAllocation Pass - This pass implements the Partitioned Boolean
  /// Quadratic Prograaming (PBQP) based register allocator.
  ///
  FunctionPass *createDefaultPBQPRegisterAllocator();

  /// PrologEpilogCodeInserter - This pass inserts prolog and epilog code,
  /// and eliminates abstract frame references.
  extern char &PrologEpilogCodeInserterID;

  /// ExpandPostRAPseudos - This pass expands pseudo instructions after
  /// register allocation.
  extern char &ExpandPostRAPseudosID;

  /// createPostRAScheduler - This pass performs post register allocation
  /// scheduling.
  extern char &PostRASchedulerID;

  /// BranchFolding - This pass performs machine code CFG based
  /// optimizations to delete branches to branches, eliminate branches to
  /// successor blocks (creating fall throughs), and eliminating branches over
  /// branches.
  extern char &BranchFolderPassID;

  /// TailDuplicate - Duplicate blocks with unconditional branches
  /// into tails of their predecessors.
  extern char &TailDuplicateID;

  /// IfConverter - This pass performs machine code if conversion.
  extern char &IfConverterID;

  /// MachineBlockPlacement - This pass places basic blocks based on branch
  /// probabilities.
  extern char &MachineBlockPlacementID;

  /// MachineBlockPlacementStats - This pass collects statistics about the
  /// basic block placement using branch probabilities and block frequency
  /// information.
  extern char &MachineBlockPlacementStatsID;

  /// Code Placement - This pass optimize code placement and aligns loop
  /// headers to target specific alignment boundary.
  extern char &CodePlacementOptID;

  /// GCLowering Pass - Performs target-independent LLVM IR transformations for
  /// highly portable strategies.
  ///
  FunctionPass *createGCLoweringPass();

  /// GCMachineCodeAnalysis - Target-independent pass to mark safe points
  /// in machine code. Must be added very late during code generation, just
  /// prior to output, and importantly after all CFG transformations (such as
  /// branch folding).
  extern char &GCMachineCodeAnalysisID;

  /// Deleter Pass - Releases GC metadata.
  ///
  FunctionPass *createGCInfoDeleter();

  /// Creates a pass to print GC metadata.
  ///
  FunctionPass *createGCInfoPrinter(raw_ostream &OS);

  /// MachineCSE - This pass performs global CSE on machine instructions.
  extern char &MachineCSEID;

  /// MachineLICM - This pass performs LICM on machine instructions.
  extern char &MachineLICMID;

  /// MachineSinking - This pass performs sinking on machine instructions.
  extern char &MachineSinkingID;

  /// MachineCopyPropagation - This pass performs copy propagation on
  /// machine instructions.
  extern char &MachineCopyPropagationID;

  /// PeepholeOptimizer - This pass performs peephole optimizations -
  /// like extension and comparison eliminations.
  extern char &PeepholeOptimizerID;

  /// OptimizePHIs - This pass optimizes machine instruction PHIs
  /// to take advantage of opportunities created during DAG legalization.
  extern char &OptimizePHIsID;

  /// StackSlotColoring - This pass performs stack slot coloring.
  extern char &StackSlotColoringID;

  /// createStackProtectorPass - This pass adds stack protectors to functions.
  ///
  FunctionPass *createStackProtectorPass(const TargetLowering *tli);

  /// createMachineVerifierPass - This pass verifies cenerated machine code
  /// instructions for correctness.
  ///
  FunctionPass *createMachineVerifierPass(const char *Banner = 0);

  /// createDwarfEHPass - This pass mulches exception handling code into a form
  /// adapted to code generation.  Required if using dwarf exception handling.
  FunctionPass *createDwarfEHPass(const TargetMachine *tm);

  /// createSjLjEHPreparePass - This pass adapts exception handling code to use
  /// the GCC-style builtin setjmp/longjmp (sjlj) to handling EH control flow.
  ///
  FunctionPass *createSjLjEHPreparePass(const TargetLowering *tli);

  /// LocalStackSlotAllocation - This pass assigns local frame indices to stack
  /// slots relative to one another and allocates base registers to access them
  /// when it is estimated by the target to be out of range of normal frame
  /// pointer or stack pointer index addressing.
  extern char &LocalStackSlotAllocationID;

  /// ExpandISelPseudos - This pass expands pseudo-instructions.
  extern char &ExpandISelPseudosID;

  /// createExecutionDependencyFixPass - This pass fixes execution time
  /// problems with dependent instructions, such as switching execution
  /// domains to match.
  ///
  /// The pass will examine instructions using and defining registers in RC.
  ///
  FunctionPass *createExecutionDependencyFixPass(const TargetRegisterClass *RC);

  /// UnpackMachineBundles - This pass unpack machine instruction bundles.
  extern char &UnpackMachineBundlesID;

  /// FinalizeMachineBundles - This pass finalize machine instruction
  /// bundles (created earlier, e.g. during pre-RA scheduling).
  extern char &FinalizeMachineBundlesID;

} // End llvm namespace

#endif
