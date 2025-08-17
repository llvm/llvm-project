//===- AutoUpgrade.h - AutoUpgrade Helpers ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  These functions are implemented by lib/IR/AutoUpgrade.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_AUTOUPGRADE_H
#define LLVM_IR_AUTOUPGRADE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include <vector>

namespace llvm {
  class AttrBuilder;
  class CallBase;
  class Constant;
  class Function;
  class Instruction;
  class GlobalVariable;
  class MDNode;
  class Module;
  class StringRef;
  class Type;
  class Value;

  template <typename T> class OperandBundleDefT;
  using OperandBundleDef = OperandBundleDefT<Value *>;

  /// This is a more granular function that simply checks an intrinsic function
  /// for upgrading, and returns true if it requires upgrading. It may return
  /// null in NewFn if the all calls to the original intrinsic function
  /// should be transformed to non-function-call instructions.
  LLVM_ABI bool
  UpgradeIntrinsicFunction(Function *F, Function *&NewFn,
                           bool CanUpgradeDebugIntrinsicsToRecords = true);

  /// This is the complement to the above, replacing a specific call to an
  /// intrinsic function with a call to the specified new function.
  LLVM_ABI void UpgradeIntrinsicCall(CallBase *CB, Function *NewFn);

  // This upgrades the comment for objc retain release markers in inline asm
  // calls
  LLVM_ABI void UpgradeInlineAsmString(std::string *AsmStr);

  /// This is an auto-upgrade hook for any old intrinsic function syntaxes
  /// which need to have both the function updated as well as all calls updated
  /// to the new function. This should only be run in a post-processing fashion
  /// so that it can update all calls to the old function.
  LLVM_ABI void UpgradeCallsToIntrinsic(Function *F);

  /// This checks for global variables which should be upgraded. If it requires
  /// upgrading, returns a pointer to the upgraded variable.
  LLVM_ABI GlobalVariable *UpgradeGlobalVariable(GlobalVariable *GV);

  /// This checks for module flags which should be upgraded. It returns true if
  /// module is modified.
  LLVM_ABI bool UpgradeModuleFlags(Module &M);

  /// Convert legacy nvvm.annotations metadata to appropriate function
  /// attributes.
  LLVM_ABI void UpgradeNVVMAnnotations(Module &M);

  /// Convert calls to ARC runtime functions to intrinsic calls and upgrade the
  /// old retain release marker to new module flag format.
  LLVM_ABI void UpgradeARCRuntime(Module &M);

  LLVM_ABI void UpgradeSectionAttributes(Module &M);

  /// Correct any IR that is relying on old function attribute behavior.
  LLVM_ABI void UpgradeFunctionAttributes(Function &F);

  /// If the given TBAA tag uses the scalar TBAA format, create a new node
  /// corresponding to the upgrade to the struct-path aware TBAA format.
  /// Otherwise return the \p TBAANode itself.
  LLVM_ABI MDNode *UpgradeTBAANode(MDNode &TBAANode);

  /// This is an auto-upgrade for bitcast between pointers with different
  /// address spaces: the instruction is replaced by a pair ptrtoint+inttoptr.
  LLVM_ABI Instruction *UpgradeBitCastInst(unsigned Opc, Value *V, Type *DestTy,
                                           Instruction *&Temp);

  /// This is an auto-upgrade for bitcast constant expression between pointers
  /// with different address spaces: the instruction is replaced by a pair
  /// ptrtoint+inttoptr.
  LLVM_ABI Constant *UpgradeBitCastExpr(unsigned Opc, Constant *C,
                                        Type *DestTy);

  /// Check the debug info version number, if it is out-dated, drop the debug
  /// info. Return true if module is modified.
  LLVM_ABI bool UpgradeDebugInfo(Module &M);

  /// Check whether a string looks like an old loop attachment tag.
  inline bool mayBeOldLoopAttachmentTag(StringRef Name) {
    return Name.starts_with("llvm.vectorizer.");
  }

  /// Upgrade the loop attachment metadata node.
  LLVM_ABI MDNode *upgradeInstructionLoopAttachment(MDNode &N);

  /// Upgrade the datalayout string by adding a section for address space
  /// pointers.
  LLVM_ABI std::string UpgradeDataLayoutString(StringRef DL, StringRef Triple);

  /// Upgrade attributes that changed format or kind.
  LLVM_ABI void UpgradeAttributes(AttrBuilder &B);

  /// Upgrade operand bundles (without knowing about their user instruction).
  LLVM_ABI void
  UpgradeOperandBundles(std::vector<OperandBundleDef> &OperandBundles);

} // End llvm namespace

#endif
