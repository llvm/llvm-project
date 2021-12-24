//===-- CFGuard.cpp - Control Flow Guard checks -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the IR transform to add Microsoft's Control Flow Guard
/// checks on Windows targets.
///
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/CFGuard.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

using OperandBundleDef = OperandBundleDefT<Value *>;

#define DEBUG_TYPE "cfguard"

STATISTIC(CFGuardCounter, "Number of Control Flow Guard checks added");

namespace {

/// Adds Control Flow Guard (CFG) checks on indirect function calls/invokes.
/// These checks ensure that the target address corresponds to the start of an
/// address-taken function. X86_64 targets use the CF_Dispatch mechanism. X86,
/// ARM, and AArch64 targets use the CF_Check machanism.
class CFGuard : public FunctionPass {
public:
  static char ID;

  enum Mechanism { CF_Check, CF_Dispatch };

  // Default constructor required for the INITIALIZE_PASS macro.
  CFGuard() : FunctionPass(ID) {
    initializeCFGuardPass(*PassRegistry::getPassRegistry());
    // By default, use the guard check mechanism.
    GuardMechanism = CF_Check;
  }

  // Recommended constructor used to specify the type of guard mechanism.
  CFGuard(Mechanism Var) : FunctionPass(ID) {
    initializeCFGuardPass(*PassRegistry::getPassRegistry());
    GuardMechanism = Var;
  }

  /// Inserts a Control Flow Guard (CFG) check on an indirect call using the CFG
  /// check mechanism. When the image is loaded, the loader puts the appropriate
  /// guard check function pointer in the __guard_check_icall_fptr global
  /// symbol. This checks that the target address is a valid address-taken
  /// function. The address of the target function is passed to the guard check
  /// function in an architecture-specific register (e.g. ECX on 32-bit X86,
  /// X15 on Aarch64, and R0 on ARM). The guard check function has no return
  /// value (if the target is invalid, the guard check funtion will raise an
  /// error).
  ///
  /// For example, the following LLVM IR:
  /// \code
  ///   %func_ptr = alloca i32 ()*, align 8
  ///   store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  ///   %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  ///   %1 = call i32 %0()
  /// \endcode
  ///
  /// is transformed to:
  /// \code
  ///   %func_ptr = alloca i32 ()*, align 8
  ///   store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  ///   %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  ///   %1 = load void (i8*)*, void (i8*)** @__guard_check_icall_fptr
  ///   %2 = bitcast i32 ()* %0 to i8*
  ///   call cfguard_checkcc void %1(i8* %2)
  ///   %3 = call i32 %0()
  /// \endcode
  ///
  /// For example, the following X86 assembly code:
  /// \code
  ///   movl  $_target_func, %eax
  ///   calll *%eax
  /// \endcode
  ///
  /// is transformed to:
  /// \code
  /// 	movl	$_target_func, %ecx
  /// 	calll	*___guard_check_icall_fptr
  /// 	calll	*%ecx
  /// \endcode
  ///
  /// \param CB indirect call to instrument.
  void insertCFGuardCheck(CallBase *CB);

  /// Inserts a Control Flow Guard (CFG) check on an indirect call using the CFG
  /// dispatch mechanism. When the image is loaded, the loader puts the
  /// appropriate guard check function pointer in the
  /// __guard_dispatch_icall_fptr global symbol. This checks that the target
  /// address is a valid address-taken function and, if so, tail calls the
  /// target. The target address is passed in an architecture-specific register
  /// (e.g. RAX on X86_64), with all other arguments for the target function
  /// passed as usual.
  ///
  /// For example, the following LLVM IR:
  /// \code
  ///   %func_ptr = alloca i32 ()*, align 8
  ///   store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  ///   %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  ///   %1 = call i32 %0()
  /// \endcode
  ///
  /// is transformed to:
  /// \code
  ///   %func_ptr = alloca i32 ()*, align 8
  ///   store i32 ()* @target_func, i32 ()** %func_ptr, align 8
  ///   %0 = load i32 ()*, i32 ()** %func_ptr, align 8
  ///   %1 = load i32 ()*, i32 ()** @__guard_dispatch_icall_fptr
  ///   %2 = call i32 %1() [ "cfguardtarget"(i32 ()* %0) ]
  /// \endcode
  ///
  /// For example, the following X86_64 assembly code:
  /// \code
  ///   leaq   target_func(%rip), %rax
  ///	  callq  *%rax
  /// \endcode
  ///
  /// is transformed to:
  /// \code
  ///   leaq   target_func(%rip), %rax
  ///   callq  *__guard_dispatch_icall_fptr(%rip)
  /// \endcode
  ///
  /// \param CB indirect call to instrument.
  void insertCFGuardDispatch(CallBase *CB);

  bool doInitialization(Module &M) override;
  bool runOnFunction(Function &F) override;

private:
  // Only add checks if the module has the cfguard=2 flag.
  int cfguard_module_flag = 0;
  Mechanism GuardMechanism = CF_Check;
  FunctionType *GuardFnType = nullptr;
  PointerType *GuardFnPtrType = nullptr;
  Constant *GuardFnGlobal = nullptr;
};

} // end anonymous namespace

void CFGuard::insertCFGuardCheck(CallBase *CB) {

  assert(Triple(CB->getModule()->getTargetTriple()).isOSWindows() &&
         "Only applicable for Windows targets");
  assert(CB->isIndirectCall() &&
         "Control Flow Guard checks can only be added to indirect calls");

  IRBuilder<> B(CB);
  Value *CalledOperand = CB->getCalledOperand();

  // If the indirect call is called within catchpad or cleanuppad,
  // we need to copy "funclet" bundle of the call.
  SmallVector<llvm::OperandBundleDef, 1> Bundles;
  if (auto Bundle = CB->getOperandBundle(LLVMContext::OB_funclet))
    Bundles.push_back(OperandBundleDef(*Bundle));

  // Load the global symbol as a pointer to the check function.
  LoadInst *GuardCheckLoad = B.CreateLoad(GuardFnPtrType, GuardFnGlobal);

  // Create new call instruction. The CFGuard check should always be a call,
  // even if the original CallBase is an Invoke or CallBr instruction.
  CallInst *GuardCheck =
      B.CreateCall(GuardFnType, GuardCheckLoad,
                   {B.CreateBitCast(CalledOperand, B.getInt8PtrTy())}, Bundles);

  // Ensure that the first argument is passed in the correct register
  // (e.g. ECX on 32-bit X86 targets).
  GuardCheck->setCallingConv(CallingConv::CFGuard_Check);
}

void CFGuard::insertCFGuardDispatch(CallBase *CB) {

  assert(Triple(CB->getModule()->getTargetTriple()).isOSWindows() &&
         "Only applicable for Windows targets");
  assert(CB->isIndirectCall() &&
         "Control Flow Guard checks can only be added to indirect calls");

  IRBuilder<> B(CB);
  Value *CalledOperand = CB->getCalledOperand();
  Type *CalledOperandType = CalledOperand->getType();

  // Cast the guard dispatch global to the type of the called operand.
  PointerType *PTy = PointerType::get(CalledOperandType, 0);
  if (GuardFnGlobal->getType() != PTy)
    GuardFnGlobal = ConstantExpr::getBitCast(GuardFnGlobal, PTy);

  // Load the global as a pointer to a function of the same type.
  LoadInst *GuardDispatchLoad = B.CreateLoad(CalledOperandType, GuardFnGlobal);

  // Add the original call target as a cfguardtarget operand bundle.
  SmallVector<llvm::OperandBundleDef, 1> Bundles;
  CB->getOperandBundlesAsDefs(Bundles);
  Bundles.emplace_back("cfguardtarget", CalledOperand);

  // Create a copy of the call/invoke instruction and add the new bundle.
  assert((isa<CallInst>(CB) || isa<InvokeInst>(CB)) &&
         "Unknown indirect call type");
  CallBase *NewCB = CallBase::Create(CB, Bundles, CB);

  // Change the target of the call to be the guard dispatch function.
  NewCB->setCalledOperand(GuardDispatchLoad);

  // Replace the original call/invoke with the new instruction.
  CB->replaceAllUsesWith(NewCB);

  // Delete the original call/invoke.
  CB->eraseFromParent();
}

bool CFGuard::doInitialization(Module &M) {

  // Check if this module has the cfguard flag and read its value.
  if (auto *MD =
          mdconst::extract_or_null<ConstantInt>(M.getModuleFlag("cfguard")))
    cfguard_module_flag = MD->getZExtValue();

  // Skip modules for which CFGuard checks have been disabled.
  if (cfguard_module_flag != 2)
    return false;

  // Set up prototypes for the guard check and dispatch functions.
  GuardFnType = FunctionType::get(Type::getVoidTy(M.getContext()),
                                  {Type::getInt8PtrTy(M.getContext())}, false);
  GuardFnPtrType = PointerType::get(GuardFnType, 0);

  // Get or insert the guard check or dispatch global symbols.
  if (GuardMechanism == CF_Check) {
    GuardFnGlobal =
        M.getOrInsertGlobal("__guard_check_icall_fptr", GuardFnPtrType);
  } else {
    assert(GuardMechanism == CF_Dispatch && "Invalid CFGuard mechanism");
    GuardFnGlobal =
        M.getOrInsertGlobal("__guard_dispatch_icall_fptr", GuardFnPtrType);
  }

  return true;
}

bool CFGuard::runOnFunction(Function &F) {

  // Skip modules for which CFGuard checks have been disabled.
  if (cfguard_module_flag != 2)
    return false;

  SmallVector<CallBase *, 8> IndirectCalls;

  // Iterate over the instructions to find all indirect call/invoke/callbr
  // instructions. Make a separate list of pointers to indirect
  // call/invoke/callbr instructions because the original instructions will be
  // deleted as the checks are added.
  for (BasicBlock &BB : F.getBasicBlockList()) {
    for (Instruction &I : BB.getInstList()) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (CB && CB->isIndirectCall() && !CB->hasFnAttr("guard_nocf")) {
        IndirectCalls.push_back(CB);
        CFGuardCounter++;
      }
    }
  }

  // If no checks are needed, return early.
  if (IndirectCalls.empty()) {
    return false;
  }

  // For each indirect call/invoke, add the appropriate dispatch or check.
  if (GuardMechanism == CF_Dispatch) {
    for (CallBase *CB : IndirectCalls) {
      insertCFGuardDispatch(CB);
    }
  } else {
    for (CallBase *CB : IndirectCalls) {
      insertCFGuardCheck(CB);
    }
  }

  return true;
}

char CFGuard::ID = 0;
INITIALIZE_PASS(CFGuard, "CFGuard", "CFGuard", false, false)

FunctionPass *llvm::createCFGuardCheckPass() {
  return new CFGuard(CFGuard::CF_Check);
}

FunctionPass *llvm::createCFGuardDispatchPass() {
  return new CFGuard(CFGuard::CF_Dispatch);
}
