//===-- Passes.h - Target independent code generation passes ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces to access the target independent code generation
// passes provided by the LLVM backend.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PASSES_H
#define LLVM_CODEGEN_PASSES_H

#include "llvm/CodeGen/RegAllocCommon.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Discriminator.h"

#include <functional>
#include <string>

namespace llvm {

class FunctionPass;
class MachineFunction;
class MachineFunctionPass;
class ModulePass;
class Pass;
class TargetMachine;
class raw_ostream;
enum class RunOutliner;

template <typename T> class IntrusiveRefCntPtr;
namespace vfs {
class FileSystem;
} // namespace vfs

} // namespace llvm

// List of target independent CodeGen pass IDs.
namespace llvm {

/// AtomicExpandPass - At IR level this pass replace atomic instructions with
/// __atomic_* library calls, or target specific instruction which implement the
/// same semantics in a way which better fits the target backend.
LLVM_ABI FunctionPass *createAtomicExpandLegacyPass();

/// createUnreachableBlockEliminationPass - The LLVM code generator does not
/// work well with unreachable basic blocks (what live ranges make sense for a
/// block that cannot be reached?).  As such, a code generator should either
/// not instruction select unreachable blocks, or run this pass as its
/// last LLVM modifying pass to clean up blocks that are not reachable from
/// the entry block.
LLVM_ABI FunctionPass *createUnreachableBlockEliminationPass();

/// createGCEmptyBasicblocksPass - Empty basic blocks (basic blocks without
/// real code) appear as the result of optimization passes removing
/// instructions. These blocks confuscate profile analysis (e.g., basic block
/// sections) since they will share the address of their fallthrough blocks.
/// This pass garbage-collects such basic blocks.
LLVM_ABI MachineFunctionPass *createGCEmptyBasicBlocksPass();

/// createBasicBlockSections Pass - This pass assigns sections to machine
/// basic blocks and is enabled with -fbasic-block-sections.
LLVM_ABI MachineFunctionPass *createBasicBlockSectionsPass();

LLVM_ABI MachineFunctionPass *createBasicBlockPathCloningPass();

/// createMachineFunctionSplitterPass - This pass splits machine functions
/// using profile information.
LLVM_ABI MachineFunctionPass *createMachineFunctionSplitterPass();

/// createStaticDataSplitterPass - This is a machine-function pass that
/// categorizes static data hotness using profile information.
LLVM_ABI MachineFunctionPass *createStaticDataSplitterPass();

/// createStaticDataAnnotatorPASS - This is a module pass that reads from
/// StaticDataProfileInfoWrapperPass and annotates the section prefix of
/// global variables.
LLVM_ABI ModulePass *createStaticDataAnnotatorPass();

/// MachineFunctionPrinter pass - This pass prints out the machine function to
/// the given stream as a debugging tool.
LLVM_ABI MachineFunctionPass *
createMachineFunctionPrinterPass(raw_ostream &OS,
                                 const std::string &Banner = "");

/// StackFramePrinter pass - This pass prints out the machine function's
/// stack frame to the given stream as a debugging tool.
LLVM_ABI MachineFunctionPass *createStackFrameLayoutAnalysisPass();

/// MIRPrinting pass - this pass prints out the LLVM IR into the given stream
/// using the MIR serialization format.
LLVM_ABI MachineFunctionPass *createPrintMIRPass(raw_ostream &OS);

/// This pass resets a MachineFunction when it has the FailedISel property
/// as if it was just created.
/// If EmitFallbackDiag is true, the pass will emit a
/// DiagnosticInfoISelFallback for every MachineFunction it resets.
/// If AbortOnFailedISel is true, abort compilation instead of resetting.
LLVM_ABI MachineFunctionPass *
createResetMachineFunctionPass(bool EmitFallbackDiag, bool AbortOnFailedISel);

/// createCodeGenPrepareLegacyPass - Transform the code to expose more pattern
/// matching during instruction selection.
LLVM_ABI FunctionPass *createCodeGenPrepareLegacyPass();

/// This pass implements generation of target-specific intrinsics to support
/// handling of complex number arithmetic
LLVM_ABI FunctionPass *createComplexDeinterleavingPass(const TargetMachine *TM);

/// AtomicExpandID -- Lowers atomic operations in terms of either cmpxchg
/// load-linked/store-conditional loops.
LLVM_ABI extern char &AtomicExpandID;

/// MachineLoopInfo - This pass is a loop analysis pass.
LLVM_ABI extern char &MachineLoopInfoID;

/// MachineDominators - This pass is a machine dominators analysis pass.
LLVM_ABI extern char &MachineDominatorsID;

/// MachineDominanaceFrontier - This pass is a machine dominators analysis.
LLVM_ABI extern char &MachineDominanceFrontierID;

/// MachineRegionInfo - This pass computes SESE regions for machine functions.
LLVM_ABI extern char &MachineRegionInfoPassID;

/// EdgeBundles analysis - Bundle machine CFG edges.
LLVM_ABI extern char &EdgeBundlesWrapperLegacyID;

/// LiveVariables pass - This pass computes the set of blocks in which each
/// variable is life and sets machine operand kill flags.
LLVM_ABI extern char &LiveVariablesID;

/// PHIElimination - This pass eliminates machine instruction PHI nodes
/// by inserting copy instructions.  This destroys SSA information, but is the
/// desired input for some register allocators.  This pass is "required" by
/// these register allocator like this: AU.addRequiredID(PHIEliminationID);
LLVM_ABI extern char &PHIEliminationID;

/// LiveIntervals - This analysis keeps track of the live ranges of virtual
/// and physical registers.
LLVM_ABI extern char &LiveIntervalsID;

/// LiveStacks pass. An analysis keeping track of the liveness of stack slots.
LLVM_ABI extern char &LiveStacksID;

/// TwoAddressInstruction - This pass reduces two-address instructions to
/// use two operands. This destroys SSA information but it is desired by
/// register allocators.
LLVM_ABI extern char &TwoAddressInstructionPassID;

/// ProcessImpicitDefs pass - This pass removes IMPLICIT_DEFs.
LLVM_ABI extern char &ProcessImplicitDefsID;

/// RegisterCoalescer - This pass merges live ranges to eliminate copies.
LLVM_ABI extern char &RegisterCoalescerID;

/// MachineScheduler - This pass schedules machine instructions.
LLVM_ABI extern char &MachineSchedulerID;

/// PostMachineScheduler - This pass schedules machine instructions postRA.
LLVM_ABI extern char &PostMachineSchedulerID;

/// SpillPlacement analysis. Suggest optimal placement of spill code between
/// basic blocks.
LLVM_ABI extern char &SpillPlacementID;

/// ShrinkWrap pass. Look for the best place to insert save and restore
// instruction and update the MachineFunctionInfo with that information.
LLVM_ABI extern char &ShrinkWrapID;

/// LiveRangeShrink pass. Move instruction close to its definition to shrink
/// the definition's live range.
LLVM_ABI extern char &LiveRangeShrinkID;

/// Greedy register allocator.
LLVM_ABI extern char &RAGreedyLegacyID;

/// Basic register allocator.
LLVM_ABI extern char &RABasicID;

/// VirtRegRewriter pass. Rewrite virtual registers to physical registers as
/// assigned in VirtRegMap.
LLVM_ABI extern char &VirtRegRewriterID;
LLVM_ABI FunctionPass *createVirtRegRewriter(bool ClearVirtRegs = true);

/// UnreachableMachineBlockElimination - This pass removes unreachable
/// machine basic blocks.
LLVM_ABI extern char &UnreachableMachineBlockElimID;

/// DeadMachineInstructionElim - This pass removes dead machine instructions.
LLVM_ABI extern char &DeadMachineInstructionElimID;

/// This pass adds dead/undef flags after analyzing subregister lanes.
LLVM_ABI extern char &DetectDeadLanesID;

/// This pass perform post-ra machine sink for COPY instructions.
LLVM_ABI extern char &PostRAMachineSinkingID;

/// This pass adds flow sensitive discriminators.
LLVM_ABI extern char &MIRAddFSDiscriminatorsID;

/// This pass reads flow sensitive profile.
LLVM_ABI extern char &MIRProfileLoaderPassID;

// This pass gives undef values a Pseudo Instruction definition for
// Instructions to ensure early-clobber is followed when using the greedy
// register allocator.
LLVM_ABI extern char &InitUndefID;

/// FastRegisterAllocation Pass - This pass register allocates as fast as
/// possible. It is best suited for debug code where live ranges are short.
///
LLVM_ABI FunctionPass *createFastRegisterAllocator();
LLVM_ABI FunctionPass *createFastRegisterAllocator(RegAllocFilterFunc F,
                                                   bool ClearVirtRegs);

/// BasicRegisterAllocation Pass - This pass implements a degenerate global
/// register allocator using the basic regalloc framework.
///
LLVM_ABI FunctionPass *createBasicRegisterAllocator();
LLVM_ABI FunctionPass *createBasicRegisterAllocator(RegAllocFilterFunc F);

/// Greedy register allocation pass - This pass implements a global register
/// allocator for optimized builds.
///
LLVM_ABI FunctionPass *createGreedyRegisterAllocator();
LLVM_ABI FunctionPass *createGreedyRegisterAllocator(RegAllocFilterFunc F);

/// PBQPRegisterAllocation Pass - This pass implements the Partitioned Boolean
/// Quadratic Prograaming (PBQP) based register allocator.
///
LLVM_ABI FunctionPass *createDefaultPBQPRegisterAllocator();

/// PrologEpilogCodeInserter - This pass inserts prolog and epilog code,
/// and eliminates abstract frame references.
LLVM_ABI extern char &PrologEpilogCodeInserterID;
LLVM_ABI MachineFunctionPass *createPrologEpilogInserterPass();

/// ExpandPostRAPseudos - This pass expands pseudo instructions after
/// register allocation.
LLVM_ABI extern char &ExpandPostRAPseudosID;

/// PostRAHazardRecognizer - This pass runs the post-ra hazard
/// recognizer.
LLVM_ABI extern char &PostRAHazardRecognizerID;

/// PostRAScheduler - This pass performs post register allocation
/// scheduling.
LLVM_ABI extern char &PostRASchedulerID;

/// BranchFolding - This pass performs machine code CFG based
/// optimizations to delete branches to branches, eliminate branches to
/// successor blocks (creating fall throughs), and eliminating branches over
/// branches.
LLVM_ABI extern char &BranchFolderPassID;

/// BranchRelaxation - This pass replaces branches that need to jump further
/// than is supported by a branch instruction.
LLVM_ABI extern char &BranchRelaxationPassID;

/// MachineFunctionPrinterPass - This pass prints out MachineInstr's.
LLVM_ABI extern char &MachineFunctionPrinterPassID;

/// MIRPrintingPass - this pass prints out the LLVM IR using the MIR
/// serialization format.
LLVM_ABI extern char &MIRPrintingPassID;

/// TailDuplicate - Duplicate blocks with unconditional branches
/// into tails of their predecessors.
LLVM_ABI extern char &TailDuplicateLegacyID;

/// Duplicate blocks with unconditional branches into tails of their
/// predecessors. Variant that works before register allocation.
LLVM_ABI extern char &EarlyTailDuplicateLegacyID;

/// MachineTraceMetrics - This pass computes critical path and CPU resource
/// usage in an ensemble of traces.
LLVM_ABI extern char &MachineTraceMetricsID;

/// EarlyIfConverter - This pass performs if-conversion on SSA form by
/// inserting cmov instructions.
LLVM_ABI extern char &EarlyIfConverterLegacyID;

/// EarlyIfPredicator - This pass performs if-conversion on SSA form by
/// predicating if/else block and insert select at the join point.
LLVM_ABI extern char &EarlyIfPredicatorID;

/// This pass performs instruction combining using trace metrics to estimate
/// critical-path and resource depth.
LLVM_ABI extern char &MachineCombinerID;

/// StackSlotColoring - This pass performs stack coloring and merging.
/// It merges disjoint allocas to reduce the stack size.
LLVM_ABI extern char &StackColoringLegacyID;

/// StackFramePrinter - This pass prints the stack frame layout and variable
/// mappings.
LLVM_ABI extern char &StackFrameLayoutAnalysisPassID;

/// IfConverter - This pass performs machine code if conversion.
LLVM_ABI extern char &IfConverterID;

LLVM_ABI FunctionPass *
createIfConverter(std::function<bool(const MachineFunction &)> Ftor);

/// MachineBlockPlacement - This pass places basic blocks based on branch
/// probabilities.
LLVM_ABI extern char &MachineBlockPlacementID;

/// MachineBlockPlacementStats - This pass collects statistics about the
/// basic block placement using branch probabilities and block frequency
/// information.
LLVM_ABI extern char &MachineBlockPlacementStatsID;

/// GCLowering Pass - Used by gc.root to perform its default lowering
/// operations.
LLVM_ABI FunctionPass *createGCLoweringPass();

/// GCLowering Pass - Used by gc.root to perform its default lowering
/// operations.
LLVM_ABI extern char &GCLoweringID;

/// ShadowStackGCLowering - Implements the custom lowering mechanism
/// used by the shadow stack GC.  Only runs on functions which opt in to
/// the shadow stack collector.
LLVM_ABI FunctionPass *createShadowStackGCLoweringPass();

/// ShadowStackGCLowering - Implements the custom lowering mechanism
/// used by the shadow stack GC.
LLVM_ABI extern char &ShadowStackGCLoweringID;

/// GCMachineCodeAnalysis - Target-independent pass to mark safe points
/// in machine code. Must be added very late during code generation, just
/// prior to output, and importantly after all CFG transformations (such as
/// branch folding).
LLVM_ABI extern char &GCMachineCodeAnalysisID;

/// MachineCSE - This pass performs global CSE on machine instructions.
LLVM_ABI extern char &MachineCSELegacyID;

/// MIRCanonicalizer - This pass canonicalizes MIR by renaming vregs
/// according to the semantics of the instruction as well as hoists
/// code.
LLVM_ABI extern char &MIRCanonicalizerID;

/// ImplicitNullChecks - This pass folds null pointer checks into nearby
/// memory operations.
LLVM_ABI extern char &ImplicitNullChecksID;

/// This pass performs loop invariant code motion on machine instructions.
LLVM_ABI extern char &MachineLICMID;

/// This pass performs loop invariant code motion on machine instructions.
/// This variant works before register allocation. \see MachineLICMID.
LLVM_ABI extern char &EarlyMachineLICMID;

/// MachineSinking - This pass performs sinking on machine instructions.
LLVM_ABI extern char &MachineSinkingLegacyID;

/// MachineCopyPropagation - This pass performs copy propagation on
/// machine instructions.
LLVM_ABI extern char &MachineCopyPropagationID;

LLVM_ABI MachineFunctionPass *
createMachineCopyPropagationPass(bool UseCopyInstr);

/// MachineLateInstrsCleanup - This pass removes redundant identical
/// instructions after register allocation and rematerialization.
LLVM_ABI extern char &MachineLateInstrsCleanupID;

/// PeepholeOptimizer - This pass performs peephole optimizations -
/// like extension and comparison eliminations.
LLVM_ABI extern char &PeepholeOptimizerLegacyID;

/// OptimizePHIs - This pass optimizes machine instruction PHIs
/// to take advantage of opportunities created during DAG legalization.
LLVM_ABI extern char &OptimizePHIsLegacyID;

/// StackSlotColoring - This pass performs stack slot coloring.
LLVM_ABI extern char &StackSlotColoringID;

/// This pass lays out funclets contiguously.
LLVM_ABI extern char &FuncletLayoutID;

/// This pass inserts the XRay instrumentation sleds if they are supported by
/// the target platform.
LLVM_ABI extern char &XRayInstrumentationID;

/// This pass inserts FEntry calls
LLVM_ABI extern char &FEntryInserterID;

/// This pass implements the "patchable-function" attribute.
LLVM_ABI extern char &PatchableFunctionID;

/// createStackProtectorPass - This pass adds stack protectors to functions.
///
LLVM_ABI FunctionPass *createStackProtectorPass();

/// createMachineVerifierPass - This pass verifies cenerated machine code
/// instructions for correctness.
///
LLVM_ABI FunctionPass *createMachineVerifierPass(const std::string &Banner);

/// createDwarfEHPass - This pass mulches exception handling code into a form
/// adapted to code generation.  Required if using dwarf exception handling.
LLVM_ABI FunctionPass *createDwarfEHPass(CodeGenOptLevel OptLevel);

/// createWinEHPass - Prepares personality functions used by MSVC on Windows,
/// in addition to the Itanium LSDA based personalities.
LLVM_ABI FunctionPass *createWinEHPass(bool DemoteCatchSwitchPHIOnly = false);

/// createSjLjEHPreparePass - This pass adapts exception handling code to use
/// the GCC-style builtin setjmp/longjmp (sjlj) to handling EH control flow.
///
LLVM_ABI FunctionPass *createSjLjEHPreparePass(const TargetMachine *TM);

/// createWasmEHPass - This pass adapts exception handling code to use
/// WebAssembly's exception handling scheme.
LLVM_ABI FunctionPass *createWasmEHPass();

/// LocalStackSlotAllocation - This pass assigns local frame indices to stack
/// slots relative to one another and allocates base registers to access them
/// when it is estimated by the target to be out of range of normal frame
/// pointer or stack pointer index addressing.
LLVM_ABI extern char &LocalStackSlotAllocationID;

/// This pass expands pseudo-instructions, reserves registers and adjusts
/// machine frame information.
LLVM_ABI extern char &FinalizeISelID;

/// UnpackMachineBundles - This pass unpack machine instruction bundles.
LLVM_ABI extern char &UnpackMachineBundlesID;

LLVM_ABI FunctionPass *
createUnpackMachineBundles(std::function<bool(const MachineFunction &)> Ftor);

/// StackMapLiveness - This pass analyses the register live-out set of
/// stackmap/patchpoint intrinsics and attaches the calculated information to
/// the intrinsic for later emission to the StackMap.
LLVM_ABI extern char &StackMapLivenessID;

// MachineSanitizerBinaryMetadata - appends/finalizes sanitizer binary
// metadata after llvm SanitizerBinaryMetadata pass.
LLVM_ABI extern char &MachineSanitizerBinaryMetadataID;

/// RemoveLoadsIntoFakeUses pass.
LLVM_ABI extern char &RemoveLoadsIntoFakeUsesID;

/// RemoveRedundantDebugValues pass.
LLVM_ABI extern char &RemoveRedundantDebugValuesID;

/// MachineCFGPrinter pass.
LLVM_ABI extern char &MachineCFGPrinterID;

/// LiveDebugValues pass
LLVM_ABI extern char &LiveDebugValuesID;

/// InterleavedAccess Pass - This pass identifies and matches interleaved
/// memory accesses to target specific intrinsics.
///
LLVM_ABI FunctionPass *createInterleavedAccessPass();

/// InterleavedLoadCombines Pass - This pass identifies interleaved loads and
/// combines them into wide loads detectable by InterleavedAccessPass
///
LLVM_ABI FunctionPass *createInterleavedLoadCombinePass();

/// LowerEmuTLS - This pass generates __emutls_[vt].xyz variables for all
/// TLS variables for the emulated TLS model.
///
LLVM_ABI ModulePass *createLowerEmuTLSPass();

/// This pass lowers the \@llvm.load.relative and \@llvm.objc.* intrinsics to
/// instructions.  This is unsafe to do earlier because a pass may combine the
/// constant initializer into the load, which may result in an overflowing
/// evaluation.
LLVM_ABI ModulePass *createPreISelIntrinsicLoweringPass();

/// GlobalMerge - This pass merges internal (by default) globals into structs
/// to enable reuse of a base pointer by indexed addressing modes.
/// It can also be configured to focus on size optimizations only.
///
LLVM_ABI Pass *
createGlobalMergePass(const TargetMachine *TM, unsigned MaximalOffset,
                      bool OnlyOptimizeForSize = false,
                      bool MergeExternalByDefault = false,
                      bool MergeConstantByDefault = false,
                      bool MergeConstAggressiveByDefault = false);

/// This pass splits the stack into a safe stack and an unsafe stack to
/// protect against stack-based overflow vulnerabilities.
LLVM_ABI FunctionPass *createSafeStackPass();

/// This pass detects subregister lanes in a virtual register that are used
/// independently of other lanes and splits them into separate virtual
/// registers.
LLVM_ABI extern char &RenameIndependentSubregsID;

/// This pass is executed POST-RA to collect which physical registers are
/// preserved by given machine function.
LLVM_ABI FunctionPass *createRegUsageInfoCollector();

/// Return a MachineFunction pass that identifies call sites
/// and propagates register usage information of callee to caller
/// if available with PysicalRegisterUsageInfo pass.
LLVM_ABI FunctionPass *createRegUsageInfoPropPass();

/// This pass performs software pipelining on machine instructions.
LLVM_ABI extern char &MachinePipelinerID;

/// This pass frees the memory occupied by the MachineFunction.
LLVM_ABI FunctionPass *createFreeMachineFunctionPass();

/// This pass performs merging similar functions globally.
LLVM_ABI ModulePass *createGlobalMergeFuncPass();

/// This pass performs outlining on machine instructions directly before
/// printing assembly.
LLVM_ABI ModulePass *createMachineOutlinerPass(RunOutliner RunOutlinerMode);

/// This pass expands the reduction intrinsics into sequences of shuffles.
LLVM_ABI FunctionPass *createExpandReductionsPass();

// This pass replaces intrinsics operating on vector operands with calls to
// the corresponding function in a vector library (e.g., SVML, libmvec).
LLVM_ABI FunctionPass *createReplaceWithVeclibLegacyPass();

// Expands large div/rem instructions.
LLVM_ABI FunctionPass *createExpandLargeDivRemPass();

// Expands large div/rem instructions.
LLVM_ABI FunctionPass *createExpandFpPass();

// This pass expands memcmp() to load/stores.
LLVM_ABI FunctionPass *createExpandMemCmpLegacyPass();

/// Creates Break False Dependencies pass. \see BreakFalseDeps.cpp
LLVM_ABI FunctionPass *createBreakFalseDeps();

// This pass expands indirectbr instructions.
LLVM_ABI FunctionPass *createIndirectBrExpandPass();

/// Creates CFI Fixup pass. \see CFIFixup.cpp
LLVM_ABI FunctionPass *createCFIFixup();

/// Creates CFI Instruction Inserter pass. \see CFIInstrInserter.cpp
LLVM_ABI FunctionPass *createCFIInstrInserter();

// Expands floating point instructions.
FunctionPass *createExpandFpPass(CodeGenOptLevel);

/// Creates CFGuard longjmp target identification pass.
/// \see CFGuardLongjmp.cpp
LLVM_ABI FunctionPass *createCFGuardLongjmpPass();

/// Creates Windows EH Continuation Guard target identification pass.
/// \see EHContGuardTargets.cpp
LLVM_ABI FunctionPass *createEHContGuardTargetsPass();

/// Create Hardware Loop pass. \see HardwareLoops.cpp
LLVM_ABI FunctionPass *createHardwareLoopsLegacyPass();

/// This pass inserts pseudo probe annotation for callsite profiling.
LLVM_ABI FunctionPass *createPseudoProbeInserter();

/// Create IR Type Promotion pass. \see TypePromotion.cpp
LLVM_ABI FunctionPass *createTypePromotionLegacyPass();

/// Add Flow Sensitive Discriminators. PassNum specifies the
/// sequence number of this pass (starting from 1).
LLVM_ABI FunctionPass *
createMIRAddFSDiscriminatorsPass(sampleprof::FSDiscriminatorPass P);

/// Read Flow Sensitive Profile.
LLVM_ABI FunctionPass *
createMIRProfileLoaderPass(std::string File, std::string RemappingFile,
                           sampleprof::FSDiscriminatorPass P,
                           IntrusiveRefCntPtr<vfs::FileSystem> FS);

/// Creates MIR Debugify pass. \see MachineDebugify.cpp
LLVM_ABI ModulePass *createDebugifyMachineModulePass();

/// Creates MIR Strip Debug pass. \see MachineStripDebug.cpp
/// If OnlyDebugified is true then it will only strip debug info if it was
/// added by a Debugify pass. The module will be left unchanged if the debug
/// info was generated by another source such as clang.
LLVM_ABI ModulePass *createStripDebugMachineModulePass(bool OnlyDebugified);

/// Creates MIR Check Debug pass. \see MachineCheckDebugify.cpp
LLVM_ABI ModulePass *createCheckDebugMachineModulePass();

/// The pass fixups statepoint machine instruction to replace usage of
/// caller saved registers with stack slots.
LLVM_ABI extern char &FixupStatepointCallerSavedID;

/// The pass transforms load/store <256 x i32> to AMX load/store intrinsics
/// or split the data to two <128 x i32>.
LLVM_ABI FunctionPass *createX86LowerAMXTypePass();

/// The pass transforms amx intrinsics to scalar operation if the function has
/// optnone attribute or it is O0.
LLVM_ABI FunctionPass *createX86LowerAMXIntrinsicsPass();

/// When learning an eviction policy, extract score(reward) information,
/// otherwise this does nothing
LLVM_ABI FunctionPass *createRegAllocScoringPass();

/// JMC instrument pass.
LLVM_ABI ModulePass *createJMCInstrumenterPass();

/// This pass converts conditional moves to conditional jumps when profitable.
LLVM_ABI FunctionPass *createSelectOptimizePass();

LLVM_ABI FunctionPass *createCallBrPass();

/// Creates Windows Secure Hot Patch pass. \see WindowsSecureHotPatching.cpp
LLVM_ABI ModulePass *createWindowsSecureHotPatchingPass();

/// Lowers KCFI operand bundles for indirect calls.
LLVM_ABI FunctionPass *createKCFIPass();
} // namespace llvm

#endif
