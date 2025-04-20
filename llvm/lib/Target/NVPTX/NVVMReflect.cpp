//===- NVVMReflect.cpp - NVVM Emulate conditional compilation -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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
#include "llvm/ADT/StringExtras.h"
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
#define NVVM_REFLECT_FUNCTION "__nvvm_reflect"
#define NVVM_REFLECT_OCL_FUNCTION "__nvvm_reflect_ocl"
// Argument of reflect call to retrive arch number
#define CUDA_ARCH_NAME "__CUDA_ARCH"
// Argument of reflect call to retrive ftz mode
#define CUDA_FTZ_NAME "__CUDA_FTZ"
// Name of module metadata where ftz mode is stored
#define CUDA_FTZ_MODULE_NAME "nvvm-reflect-ftz"

using namespace llvm;

#define DEBUG_TYPE "nvvm-reflect"

namespace {
class NVVMReflect {
  // Map from reflect function call arguments to the value to replace the call
  // with. Should include __CUDA_FTZ and __CUDA_ARCH values.
  StringMap<unsigned> ReflectMap;
  bool handleReflectFunction(Module &M, StringRef ReflectName);
  void populateReflectMap(Module &M);
  void foldReflectCall(CallInst *Call, Constant *NewValue);

public:
  // __CUDA_FTZ is assigned in `runOnModule` by checking nvvm-reflect-ftz module
  // metadata.
  explicit NVVMReflect(unsigned SmVersion)
      : ReflectMap({{CUDA_ARCH_NAME, SmVersion * 10}}) {}
  bool runOnModule(Module &M);
};

class NVVMReflectLegacyPass : public ModulePass {
  NVVMReflect Impl;

public:
  static char ID;
  NVVMReflectLegacyPass(unsigned SmVersion) : ModulePass(ID), Impl(SmVersion) {}
  bool runOnModule(Module &M) override;
};
} // namespace

ModulePass *llvm::createNVVMReflectPass(unsigned SmVersion) {
  return new NVVMReflectLegacyPass(SmVersion);
}

static cl::opt<bool>
    NVVMReflectEnabled("nvvm-reflect-enable", cl::init(true), cl::Hidden,
                       cl::desc("NVVM reflection, enabled by default"));

char NVVMReflectLegacyPass::ID = 0;
INITIALIZE_PASS(NVVMReflectLegacyPass, "nvvm-reflect",
                "Replace occurrences of __nvvm_reflect() calls with 0/1", false,
                false)

// Allow users to specify additional key/value pairs to reflect. These key/value
// pairs are the last to be added to the ReflectMap, and therefore will take
// precedence over initial values (i.e. __CUDA_FTZ from module medadata and
// __CUDA_ARCH from SmVersion).
static cl::list<std::string> ReflectList(
    "nvvm-reflect-add", cl::value_desc("name=<int>"), cl::Hidden,
    cl::desc("A key=value pair. Replace __nvvm_reflect(name) with value."),
    cl::ValueRequired);

// Set the ReflectMap with, first, the value of __CUDA_FTZ from module metadata,
// and then the key/value pairs from the command line.
void NVVMReflect::populateReflectMap(Module &M) {
  if (auto *Flag = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag(CUDA_FTZ_MODULE_NAME)))
    ReflectMap[CUDA_FTZ_NAME] = Flag->getSExtValue();

  for (auto &Option : ReflectList) {
    LLVM_DEBUG(dbgs() << "ReflectOption : " << Option << "\n");
    StringRef OptionRef(Option);
    auto [Name, Val] = OptionRef.split('=');
    if (Name.empty())
      report_fatal_error(Twine("Empty name in nvvm-reflect-add option '") +
                         Option + "'");
    if (Val.empty())
      report_fatal_error(Twine("Missing value in nvvm-reflect-add option '") +
                         Option + "'");
    unsigned ValInt;
    if (!to_integer(Val.trim(), ValInt, 10))
      report_fatal_error(
          Twine("integer value expected in nvvm-reflect-add option '") +
          Option + "'");
    ReflectMap[Name] = ValInt;
  }
}

/// Process a reflect function by finding all its calls and replacing them with
/// appropriate constant values. For __CUDA_FTZ, uses the module flag value.
/// For __CUDA_ARCH, uses SmVersion * 10. For all other strings, uses 0.
bool NVVMReflect::handleReflectFunction(Module &M, StringRef ReflectName) {
  Function *F = M.getFunction(ReflectName);
  if (!F)
    return false;
  assert(F->isDeclaration() && "_reflect function should not have a body");
  assert(F->getReturnType()->isIntegerTy() &&
         "_reflect's return type should be integer");

  const bool Changed = !F->use_empty();
  for (User *U : make_early_inc_range(F->users())) {
    // Reflect function calls look like:
    // @arch = private unnamed_addr addrspace(1) constant [12 x i8]
    // c"__CUDA_ARCH\00" call i32 @__nvvm_reflect(ptr addrspacecast (ptr
    // addrspace(1) @arch to ptr)) We need to extract the string argument from
    // the call (i.e. "__CUDA_ARCH")
    auto *Call = dyn_cast<CallInst>(U);
    if (!Call)
      report_fatal_error(
          "__nvvm_reflect can only be used in a call instruction");
    if (Call->getNumOperands() != 2)
      report_fatal_error("__nvvm_reflect requires exactly one argument");

    auto *GlobalStr =
        dyn_cast<Constant>(Call->getArgOperand(0)->stripPointerCasts());
    if (!GlobalStr)
      report_fatal_error("__nvvm_reflect argument must be a constant string");

    auto *ConstantStr =
        dyn_cast<ConstantDataSequential>(GlobalStr->getOperand(0));
    if (!ConstantStr)
      report_fatal_error("__nvvm_reflect argument must be a string constant");
    if (!ConstantStr->isCString())
      report_fatal_error(
          "__nvvm_reflect argument must be a null-terminated string");

    StringRef ReflectArg = ConstantStr->getAsString().drop_back();
    if (ReflectArg.empty())
      report_fatal_error("__nvvm_reflect argument cannot be empty");
    // Now that we have extracted the string argument, we can look it up in the
    // ReflectMap
    unsigned ReflectVal = 0; // The default value is 0
    if (ReflectMap.contains(ReflectArg))
      ReflectVal = ReflectMap[ReflectArg];

    LLVM_DEBUG(dbgs() << "Replacing call of reflect function " << F->getName()
                      << "(" << ReflectArg << ") with value " << ReflectVal
                      << "\n");
    auto *NewValue = ConstantInt::get(Call->getType(), ReflectVal);
    foldReflectCall(Call, NewValue);
    Call->eraseFromParent();
  }

  // Remove the __nvvm_reflect function from the module
  F->eraseFromParent();
  return Changed;
}

void NVVMReflect::foldReflectCall(CallInst *Call, Constant *NewValue) {
  SmallVector<Instruction *, 8> Worklist;
  // Replace an instruction with a constant and add all users of the instruction
  // to the worklist
  auto ReplaceInstructionWithConst = [&](Instruction *I, Constant *C) {
    for (auto *U : I->users())
      if (auto *UI = dyn_cast<Instruction>(U))
        Worklist.push_back(UI);
    I->replaceAllUsesWith(C);
  };

  ReplaceInstructionWithConst(Call, NewValue);

  auto &DL = Call->getModule()->getDataLayout();
  while (!Worklist.empty()) {
    auto *I = Worklist.pop_back_val();
    if (auto *C = ConstantFoldInstruction(I, DL)) {
      ReplaceInstructionWithConst(I, C);
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
  populateReflectMap(M);
  bool Changed = true;
  Changed |= handleReflectFunction(M, NVVM_REFLECT_FUNCTION);
  Changed |= handleReflectFunction(M, NVVM_REFLECT_OCL_FUNCTION);
  Changed |=
      handleReflectFunction(M, Intrinsic::getName(Intrinsic::nvvm_reflect));
  return Changed;
}

bool NVVMReflectLegacyPass::runOnModule(Module &M) {
  return Impl.runOnModule(M);
}

PreservedAnalyses NVVMReflectPass::run(Module &M, ModuleAnalysisManager &AM) {
  return NVVMReflect(SmVersion).runOnModule(M) ? PreservedAnalyses::none()
                                               : PreservedAnalyses::all();
}
