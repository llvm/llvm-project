//===- NVVMReflect.cpp - NVVM Emulate conditional compilation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// NVIDIA_COPYRIGHT_BEGIN
//
// Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA_COPYRIGHT_END
//
//===----------------------------------------------------------------------===//
//
// This pass replaces occurrences of __nvvm_reflect("foo") and llvm.nvvm.reflect
// with an integer.
//
// We choose the value we use by looking at metadata in the module itself.  Note
// that we intentionally only have one way to choose these values, because other
// parts of LLVM (particularly, InstCombineCall) rely on being able to predict
// the values chosen by this pass.
//
// If we see an unknown string, we replace its call with 0.
//
//===----------------------------------------------------------------------===//

#include "NVPTX.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/ADT/StringExtras.h"
#define NVVM_REFLECT_FUNCTION "__nvvm_reflect"
#define NVVM_REFLECT_OCL_FUNCTION "__nvvm_reflect_ocl"

using namespace llvm;

#define DEBUG_TYPE "nvvm-reflect"

namespace {
class NVVMReflect : public ModulePass {
private:
  StringMap<int> VarMap;
  void handleReflectFunction(Function *F);
  void setVarMap(Module &M);
  void foldReflectCall(CallInst *Call, Constant *NewValue);
public:
  static char ID;
  NVVMReflect() : NVVMReflect(0) {}
  // __CUDA_FTZ is assigned in `runOnModule` by checking nvvm-reflect-ftz module
  // metadata.
  explicit NVVMReflect(unsigned SmVersion) : ModulePass(ID), VarMap({{"__CUDA_ARCH", SmVersion * 10}}) {
    initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
  }
  // This mapping will contain should include __CUDA_FTZ and __CUDA_ARCH values.
  explicit NVVMReflect(const StringMap<int> &Mapping) : ModulePass(ID), VarMap(Mapping) {
    initializeNVVMReflectPass(*PassRegistry::getPassRegistry());
  }
  bool runOnModule(Module &M) override;
};
} // namespace

ModulePass *llvm::createNVVMReflectPass(unsigned int SmVersion) {
  return new NVVMReflect(SmVersion);
}

static cl::opt<bool>
    NVVMReflectEnabled("nvvm-reflect-enable", cl::init(true), cl::Hidden,
                       cl::desc("NVVM reflection, enabled by default"));

char NVVMReflect::ID = 0;
INITIALIZE_PASS(NVVMReflect, "nvvm-reflect",
                "Replace occurrences of __nvvm_reflect() calls with 0/1", false,
                false)

// Allow users to specify additional key/value pairs to reflect. These key/value pairs
// are the last to be added to the VarMap, and therefore will take precedence over initial
// values (i.e. __CUDA_FTZ from module medadata and __CUDA_ARCH from SmVersion).
static cl::list<std::string>
ReflectList("nvvm-reflect-add", cl::value_desc("name=<int>"), cl::Hidden,
            cl::desc("list of comma-separated key=value pairs"),
            cl::ValueRequired);

// Set the VarMap with, first, the value of __CUDA_FTZ from module metadata, and then
// the key/value pairs from the command line.
void NVVMReflect::setVarMap(Module &M) {
  LLVM_DEBUG(dbgs() << "Reflect list values:\n");
  for (StringRef Option : ReflectList) {
    LLVM_DEBUG(dbgs() << "  " << Option << "\n");
  }
  if (auto *Flag = mdconst::extract_or_null<ConstantInt>(
      M.getModuleFlag("nvvm-reflect-ftz")))
    VarMap["__CUDA_FTZ"] = Flag->getSExtValue();

  /// The command line can look as follows :
  /// -nvvm-reflect-add a=1,b=2 -nvvm-reflect-add c=3,d=0 -nvvm-reflect-add e=2
  /// The strings "a=1,b=2", "c=3,d=0", "e=2" are available in the
  /// ReflectList vector. First, each of ReflectList[i] is 'split'
  /// using "," as the delimiter. Then each of this part is split
  /// using "=" as the delimiter.
  for (StringRef Option : ReflectList) {
    LLVM_DEBUG(dbgs() << "ReflectOption : " << Option << "\n");
    while (!Option.empty()) {
      auto Split = Option.split(',');
      StringRef NameVal = Split.first;
      Option = Split.second;

      auto NameValPair = NameVal.split('=');
      assert(!NameValPair.first.empty() && !NameValPair.second.empty() && 
             "name=val expected");
      
      int Val;
      if (!to_integer(NameValPair.second.trim(), Val, 10))
        report_fatal_error("integer value expected");
      VarMap[NameValPair.first] = Val;
    }
  }
}

/// Process a reflect function by finding all its uses and replacing them with
/// appropriate constant values. For __CUDA_FTZ, uses the module flag value.
/// For __CUDA_ARCH, uses SmVersion * 10. For all other strings, uses 0.
void NVVMReflect::handleReflectFunction(Function *F) {
  // Validate _reflect function
  assert(F->isDeclaration() && "_reflect function should not have a body");
  assert(F->getReturnType()->isIntegerTy() && "_reflect's return type should be integer");


  // Go through the uses of the reflect function. Each use should be a CallInst
  // with a ConstantArray argument. Replace the uses with the appropriate constant values.

  // The IR for __nvvm_reflect calls differs between CUDA versions.
  //
  // CUDA 6.5 and earlier uses this sequence:
  //    %ptr = tail call i8* @llvm.nvvm.ptr.constant.to.gen.p0i8.p4i8
  //        (i8 addrspace(4)* getelementptr inbounds
  //           ([8 x i8], [8 x i8] addrspace(4)* @str, i32 0, i32 0))
  //    %reflect = tail call i32 @__nvvm_reflect(i8* %ptr)
  //
  // The value returned by Sym->getOperand(0) is a Constant with a
  // ConstantDataSequential operand which can be converted to string and used
  // for lookup.
  //
  // CUDA 7.0 does it slightly differently:
  //   %reflect = call i32 @__nvvm_reflect(i8* addrspacecast
  //        (i8 addrspace(1)* getelementptr inbounds
  //           ([8 x i8], [8 x i8] addrspace(1)* @str, i32 0, i32 0) to i8*))
  //
  // In this case, we get a Constant with a GlobalVariable operand and we need
  // to dig deeper to find its initializer with the string we'll use for lookup.

  for (User *U : make_early_inc_range(F->users())) {
    assert(isa<CallInst>(U) && "Only a call instruction can use _reflect");
    CallInst *Call = cast<CallInst>(U);

    // FIXME: Improve error handling here and elsewhere in this pass.
    assert(Call->getNumOperands() == 2 &&
           "Wrong number of operands to __nvvm_reflect function");

    // In cuda 6.5 and earlier, we will have an extra constant-to-generic
    // conversion of the string.
    const Value *Str = Call->getArgOperand(0);
    if (const CallInst *ConvCall = dyn_cast<CallInst>(Str)) {
      // Verify this is the constant-to-generic intrinsic
      Function *Callee = ConvCall->getCalledFunction();
      assert(Callee && Callee->isIntrinsic() && 
             Callee->getName().starts_with("llvm.nvvm.ptr.constant.to.gen") &&
             "Expected llvm.nvvm.ptr.constant.to.gen intrinsic");
      assert(ConvCall->getNumOperands() == 2 && "Expected one argument for ptr conversion");
      Str = ConvCall->getArgOperand(0);
    }
    // Pre opaque pointers we have a constant expression wrapping the constant
    Str = Str->stripPointerCasts();
    assert(isa<Constant>(Str) && "Format of __nvvm_reflect function not recognized");

    const Value *Operand = cast<Constant>(Str)->getOperand(0);
    if (const GlobalVariable *GV = dyn_cast<GlobalVariable>(Operand)) {
      // For CUDA-7.0 style __nvvm_reflect calls, we need to find the operand's
      // initializer.
      assert(GV->hasInitializer() && "Format of _reflect function not recognized");
      const Constant *Initializer = GV->getInitializer();
      Operand = Initializer;
    }

    assert(isa<ConstantDataSequential>(Operand) &&
           "Format of _reflect function not recognized");
    assert(cast<ConstantDataSequential>(Operand)->isCString() &&
           "Format of _reflect function not recognized");

    StringRef ReflectArg = cast<ConstantDataSequential>(Operand)->getAsString();
    // Remove the null terminator from the string
    ReflectArg = ReflectArg.substr(0, ReflectArg.size() - 1);

    int ReflectVal = 0; // The default value is 0
    if (VarMap.contains(ReflectArg)) {
      ReflectVal = VarMap[ReflectArg];
    }
    LLVM_DEBUG(dbgs() << "Replacing call of reflect function " << F->getName() << "(" << ReflectArg << ") with value " << ReflectVal << "\n");
    Constant *NewValue = ConstantInt::get(Call->getType(), ReflectVal);
    foldReflectCall(Call, NewValue);
    Call->eraseFromParent();
  }

  // Remove the __nvvm_reflect function from the module
  F->eraseFromParent();
}

void NVVMReflect::foldReflectCall(CallInst *Call, Constant *NewValue) {
  // Initialize worklist with all users of the call
  SmallVector<Instruction*, 8> Worklist;
  for (User *U : Call->users()) {
    if (Instruction *I = dyn_cast<Instruction>(U)) {
      Worklist.push_back(I);
    }
  }

  Call->replaceAllUsesWith(NewValue);

  while (!Worklist.empty()) {
    Instruction *I = Worklist.pop_back_val();
    if (Constant *C = ConstantFoldInstruction(I, Call->getModule()->getDataLayout())) {
      // Add all users of this instruction to the worklist, replace it with the constant
      // then delete it if it's dead
      for (User *U : I->users()) {
        if (Instruction *UI = dyn_cast<Instruction>(U))
          Worklist.push_back(UI);
      }
      I->replaceAllUsesWith(C);
      if (isInstructionTriviallyDead(I))
        I->eraseFromParent();
    } else if (I->isTerminator()) {
      ConstantFoldTerminator(I->getParent());
    }
  }
}

bool NVVMReflect::runOnModule(Module &M) {
  if (!NVVMReflectEnabled)
    return false;

  setVarMap(M);

  bool Changed = false;
  // Names of reflect function to find and replace
  SmallVector<StringRef, 5> ReflectNames = {
      NVVM_REFLECT_FUNCTION,
      NVVM_REFLECT_OCL_FUNCTION,
      Intrinsic::getName(Intrinsic::nvvm_reflect),
  };

  // Process all reflect functions
  for (StringRef Name : ReflectNames) {
    if (Function *ReflectFunction = M.getFunction(Name)) {
      // If the reflect functition is called, we need to replace the call
      // with the appropriate constant, modifying the IR.
      Changed |= ReflectFunction->getNumUses() > 0;
      handleReflectFunction(ReflectFunction);
    }
  }

  return Changed;
}

// Implementations for the pass that works with the new pass manager.
NVVMReflectPass::NVVMReflectPass(unsigned SmVersion) {
  VarMap["__CUDA_ARCH"] = SmVersion * 10;
}

PreservedAnalyses NVVMReflectPass::run(Module &M,
                                    ModuleAnalysisManager &AM) {
  return NVVMReflect(VarMap).runOnModule(M) ? PreservedAnalyses::none()
                                   : PreservedAnalyses::all();
}