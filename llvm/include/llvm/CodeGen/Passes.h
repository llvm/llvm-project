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

#include <iosfwd>
#include <string>

namespace llvm {

  class FunctionPass;
  class PassInfo;
  class TargetMachine;
  class TargetLowering;
  class RegisterCoalescer;

  /// createUnreachableBlockEliminationPass - The LLVM code generator does not
  /// work well with unreachable basic blocks (what live ranges make sense for a
  /// block that cannot be reached?).  As such, a code generator should either
  /// not instruction select unreachable blocks, or it can run this pass as it's
  /// last LLVM modifying pass to clean up blocks that are not reachable from
  /// the entry block.
  FunctionPass *createUnreachableBlockEliminationPass();

  /// MachineFunctionPrinter pass - This pass prints out the machine function to
  /// standard error, as a debugging tool.
  FunctionPass *createMachineFunctionPrinterPass(std::ostream *OS,
                                                 const std::string &Banner ="");

  /// MachineLoopInfo pass - This pass is a loop analysis pass.
  /// 
  extern const PassInfo *const MachineLoopInfoID;

  /// MachineDominators pass - This pass is a machine dominators analysis pass.
  /// 
  extern const PassInfo *const MachineDominatorsID;

  /// PHIElimination pass - This pass eliminates machine instruction PHI nodes
  /// by inserting copy instructions.  This destroys SSA information, but is the
  /// desired input for some register allocators.  This pass is "required" by
  /// these register allocator like this: AU.addRequiredID(PHIEliminationID);
  ///
  extern const PassInfo *const PHIEliminationID;
  
  /// StrongPHIElimination pass - This pass eliminates machine instruction PHI
  /// nodes by inserting copy instructions.  This destroys SSA information, but
  /// is the desired input for some register allocators.  This pass is
  /// "required" by these register allocator like this:
  ///    AU.addRequiredID(PHIEliminationID);
  ///  This pass is still in development
  extern const PassInfo *const StrongPHIEliminationID;

  extern const PassInfo *const PreAllocSplittingID;

  /// SimpleRegisterCoalescing pass.  Aggressively coalesces every register
  /// copy it can.
  ///
  extern const PassInfo *const SimpleRegisterCoalescingID;

  /// TwoAddressInstruction pass - This pass reduces two-address instructions to
  /// use two operands. This destroys SSA information but it is desired by
  /// register allocators.
  extern const PassInfo *const TwoAddressInstructionPassID;

  /// UnreachableMachineBlockElimination pass - This pass removes unreachable
  /// machine basic blocks.
  extern const PassInfo *const UnreachableMachineBlockElimID;

  /// DeadMachineInstructionElim pass - This pass removes dead machine
  /// instructions.
  ///
  FunctionPass *createDeadMachineInstructionElimPass();

  /// Creates a register allocator as the user specified on the command line.
  ///
  FunctionPass *createRegisterAllocator();

  /// SimpleRegisterAllocation Pass - This pass converts the input machine code
  /// from SSA form to use explicit registers by spilling every register.  Wow,
  /// great policy huh?
  ///
  FunctionPass *createSimpleRegisterAllocator();

  /// LocalRegisterAllocation Pass - This pass register allocates the input code
  /// a basic block at a time, yielding code better than the simple register
  /// allocator, but not as good as a global allocator.
  ///
  FunctionPass *createLocalRegisterAllocator();

  /// BigBlockRegisterAllocation Pass - The BigBlock register allocator
  /// munches single basic blocks at a time, like the local register
  /// allocator.  While the BigBlock allocator is a little slower, and uses
  /// somewhat more memory than the local register allocator, it tends to
  /// yield the best allocations (of any of the allocators) for blocks that
  /// have hundreds or thousands of instructions in sequence.
  ///
  FunctionPass *createBigBlockRegisterAllocator();

  /// LinearScanRegisterAllocation Pass - This pass implements the linear scan
  /// register allocation algorithm, a global register allocator.
  ///
  FunctionPass *createLinearScanRegisterAllocator();

  /// PBQPRegisterAllocation Pass - This pass implements the Partitioned Boolean
  /// Quadratic Prograaming (PBQP) based register allocator.
  ///
  FunctionPass *createPBQPRegisterAllocator();

  /// SimpleRegisterCoalescing Pass - Coalesce all copies possible.  Can run
  /// independently of the register allocator.
  ///
  RegisterCoalescer *createSimpleRegisterCoalescer();

  /// PrologEpilogCodeInserter Pass - This pass inserts prolog and epilog code,
  /// and eliminates abstract frame references.
  ///
  FunctionPass *createPrologEpilogCodeInserter();
  
  /// LowerSubregs Pass - This pass lowers subregs to register-register copies
  /// which yields suboptimal, but correct code if the register allocator
  /// cannot coalesce all subreg operations during allocation.
  ///
  FunctionPass *createLowerSubregsPass();

  /// createPostRAScheduler - under development.
  FunctionPass *createPostRAScheduler();

  /// BranchFolding Pass - This pass performs machine code CFG based
  /// optimizations to delete branches to branches, eliminate branches to
  /// successor blocks (creating fall throughs), and eliminating branches over
  /// branches.
  FunctionPass *createBranchFoldingPass(bool DefaultEnableTailMerge);

  /// IfConverter Pass - This pass performs machine code if conversion.
  FunctionPass *createIfConverterPass();

  /// LoopAligner Pass - This pass aligns loop headers to target specific
  /// alignment boundary.
  FunctionPass *createLoopAlignerPass();

  /// DebugLabelFoldingPass - This pass prunes out redundant debug labels.  This
  /// allows a debug emitter to determine if the range of two labels is empty,
  /// by seeing if the labels map to the same reduced label.
  FunctionPass *createDebugLabelFoldingPass();

  /// MachineCodeDeletion Pass - This pass deletes all of the machine code for
  /// the current function, which should happen after the function has been
  /// emitted to a .s file or to memory.
  FunctionPass *createMachineCodeDeleter();

  /// getRegisterAllocator - This creates an instance of the register allocator
  /// for the Sparc.
  FunctionPass *getRegisterAllocator(TargetMachine &T);

  /// IntrinsicLowering Pass - Performs target-independent LLVM IR
  /// transformations for highly portable strategies.
  FunctionPass *createGCLoweringPass();
  
  /// MachineCodeAnalysis Pass - Target-independent pass to mark safe points in
  /// machine code. Must be added very late during code generation, just prior
  /// to output, and importantly after all CFG transformations (such as branch
  /// folding).
  FunctionPass *createGCMachineCodeAnalysisPass();
  
  /// Deleter Pass - Releases GC metadata.
  /// 
  FunctionPass *createGCInfoDeleter();
  
  /// Creates a pass to print GC metadata.
  /// 
  FunctionPass *createGCInfoPrinter(std::ostream &OS);
  
  /// createMachineLICMPass - This pass performs LICM on machine instructions.
  /// 
  FunctionPass *createMachineLICMPass();

  /// createMachineSinkingPass - This pass performs sinking on machine
  /// instructions.
  FunctionPass *createMachineSinkingPass();

  /// createStackSlotColoringPass - This pass performs stack slot coloring.
  FunctionPass *createStackSlotColoringPass();

  /// createStackProtectorPass - This pass adds stack protectors to functions.
  FunctionPass *createStackProtectorPass(const TargetLowering *tli);

} // End llvm namespace

#endif
