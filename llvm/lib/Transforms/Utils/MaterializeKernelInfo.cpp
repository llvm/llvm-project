//===- MaterializeKernelInfo.cpp - Materialize kernel info ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass materializes compile-time kernel information as IR globals so it is
// available at runtime, for example to support profiling.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/MaterializeKernelInfo.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Transforms/Utils/KernelArgInfo.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include <cstdint>
#include <string>

using namespace llvm;

namespace {

std::string getKernelInfoSymbolName(const Function &F) {
  return (F.getName() + "_kernel_info").str();
}

KernelArgInfo getKernelArgInfo(Type *ArgTy) {
  if (ArgTy->isIntegerTy())
    return KernelArgInfo::getIntegerTy(cast<IntegerType>(ArgTy)->getBitWidth());
  if (ArgTy->isFloatTy())
    return KernelArgInfo::getFloatTy();
  if (ArgTy->isDoubleTy())
    return KernelArgInfo::getDoubleTy();
  if (ArgTy->isPointerTy())
    return KernelArgInfo::getPointerTy();
  return KernelArgInfo::getUnknownTy();
}

GlobalVariable *createKernelInfoGlobal(Module &M, StringRef Name,
                                       ArrayRef<uint8_t> EncodedKernelInfo) {
  Constant *Init = ConstantDataArray::get(M.getContext(), EncodedKernelInfo);
  auto *Ty =
      ArrayType::get(Type::getInt8Ty(M.getContext()), EncodedKernelInfo.size());
  return new GlobalVariable(M, Ty, /*isConstant=*/true,
                            GlobalValue::ExternalLinkage, Init, Name);
}

bool materializeKernelInfo(Function &F) {
  // FIXME: does not work when target is CPU. we could add a flag to the
  // module to know if it is going through the offloading driver.
  if (!F.hasKernelCallingConv())
    return false;

  const std::string KernelInfoSymbol = getKernelInfoSymbolName(F);
  if (F.getParent()->getNamedValue(KernelInfoSymbol))
    return false;

  SmallVector<KernelArgInfo::EncodeType, 16> EncodedKernelInfo;
  for (const Argument &Arg : F.args())
    EncodedKernelInfo.emplace_back(
        getKernelArgInfo(Arg.getType()).getEncodedLE());

  GlobalVariable *GV = createKernelInfoGlobal(
      *F.getParent(), KernelInfoSymbol,
      ArrayRef<uint8_t>(reinterpret_cast<uint8_t *>(EncodedKernelInfo.data()),
                        EncodedKernelInfo.size_in_bytes()));
  appendToCompilerUsed(*F.getParent(), {GV});
  return true;
}

} // namespace

PreservedAnalyses MaterializeKernelInfoPass::run(Module &M,
                                                 ModuleAnalysisManager &) {
  bool Changed = false;
  for (Function &F : M)
    Changed |= materializeKernelInfo(F);

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
