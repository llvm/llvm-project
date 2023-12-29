//===- StackProtector.h - Stack Protector Insertion -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass inserts stack protectors into functions which need them. A variable
// with a random value in it is stored onto the stack before the local variables
// are allocated. Upon exiting the block, the stored value is checked. If it's
// changed, then there was some sort of violation and the program aborts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_STACKPROTECTOR_H
#define LLVM_CODEGEN_STACKPROTECTOR_H

#include "llvm/Analysis/DomTreeUpdater.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/TargetParser/Triple.h"

namespace llvm {

class BasicBlock;
class Function;
class Module;
class TargetLoweringBase;
class TargetMachine;

class StackProtector : public FunctionPass {
private:
  static constexpr unsigned DefaultSSPBufferSize = 8;

  /// A mapping of AllocaInsts to their required SSP layout.
  using SSPLayoutMap = DenseMap<const AllocaInst *,
                                MachineFrameInfo::SSPLayoutKind>;

  const TargetMachine *TM = nullptr;

  /// TLI - Keep a pointer of a TargetLowering to consult for determining
  /// target type sizes.
  const TargetLoweringBase *TLI = nullptr;
  Triple Trip;

  Function *F = nullptr;
  Module *M = nullptr;

  std::optional<DomTreeUpdater> DTU;

  /// Layout - Mapping of allocations to the required SSPLayoutKind.
  /// StackProtector analysis will update this map when determining if an
  /// AllocaInst triggers a stack protector.
  SSPLayoutMap Layout;

  /// The minimum size of buffers that will receive stack smashing
  /// protection when -fstack-protection is used.
  unsigned SSPBufferSize = DefaultSSPBufferSize;

  // A prologue is generated.
  bool HasPrologue = false;

  // IR checking code is generated.
  bool HasIRCheck = false;

  /// InsertStackProtectors - Insert code into the prologue and epilogue of
  /// the function.
  ///
  ///  - The prologue code loads and stores the stack guard onto the stack.
  ///  - The epilogue checks the value stored in the prologue against the
  ///    original value. It calls __stack_chk_fail if they differ.
  bool InsertStackProtectors();

  /// CreateFailBB - Create a basic block to jump to when the stack protector
  /// check fails.
  BasicBlock *CreateFailBB();

public:
  static char ID; // Pass identification, replacement for typeid.

  StackProtector();

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  // Return true if StackProtector is supposed to be handled by SelectionDAG.
  bool shouldEmitSDCheck(const BasicBlock &BB) const;

  bool runOnFunction(Function &Fn) override;

  void copyToMachineFrameInfo(MachineFrameInfo &MFI) const;

  /// Check whether or not \p F needs a stack protector based upon the stack
  /// protector level.
  static bool requiresStackProtector(Function *F, SSPLayoutMap *Layout = nullptr);

};

} // end namespace llvm

#endif // LLVM_CODEGEN_STACKPROTECTOR_H
