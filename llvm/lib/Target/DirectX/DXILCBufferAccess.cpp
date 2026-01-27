//===- DXILCBufferAccess.cpp - Translate CBuffer Loads --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "DXILCBufferAccess.h"
#include "DirectX.h"
#include "llvm/Analysis/DXILResource.h"
#include "llvm/Frontend/HLSL/CBuffer.h"
#include "llvm/Frontend/HLSL/HLSLResource.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsDirectX.h"
#include "llvm/IR/ReplaceConstant.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Transforms/Utils/Local.h"

#define DEBUG_TYPE "dxil-cbuffer-access"
using namespace llvm;

static void replaceUsersOfGlobal(GlobalVariable *Global,
                                 GlobalVariable *HandleGV, size_t Offset) {
  for (Use &U : make_early_inc_range(Global->uses())) {
    auto UseInst = dyn_cast<Instruction>(U.getUser());
    // TODO: Constants? Metadata?
    assert(UseInst && "Non-instruction use of cbuffer");

    IRBuilder<> Builder(UseInst);
    LoadInst *Handle = Builder.CreateLoad(HandleGV->getValueType(), HandleGV,
                                          HandleGV->getName());
    Value *Ptr = Builder.CreateIntrinsic(
        Global->getType(), Intrinsic::dx_resource_getpointer,
        ArrayRef<Value *>{Handle,
                          ConstantInt::get(Builder.getInt32Ty(), Offset)});
    U.set(Ptr);
  }

  Global->removeFromParent();
}

static bool replaceCBufferAccesses(Module &M) {
  std::optional<hlsl::CBufferMetadata> CBufMD = hlsl::CBufferMetadata::get(
      M, [](Type *Ty) { return isa<llvm::dxil::PaddingExtType>(Ty); });
  if (!CBufMD)
    return false;

  SmallVector<Constant *> CBufferGlobals;
  for (const hlsl::CBufferMapping &Mapping : *CBufMD)
    for (const hlsl::CBufferMember &Member : Mapping.Members)
      CBufferGlobals.push_back(Member.GV);
  convertUsersOfConstantsToInstructions(CBufferGlobals);

  for (const hlsl::CBufferMapping &Mapping : *CBufMD)
    for (const hlsl::CBufferMember &Member : Mapping.Members)
      replaceUsersOfGlobal(Member.GV, Mapping.Handle, Member.Offset);

  CBufMD->eraseFromModule();
  return true;
}

PreservedAnalyses DXILCBufferAccess::run(Module &M, ModuleAnalysisManager &AM) {
  PreservedAnalyses PA;
  bool Changed = replaceCBufferAccesses(M);

  if (!Changed)
    return PreservedAnalyses::all();
  return PA;
}

namespace {
class DXILCBufferAccessLegacy : public ModulePass {
public:
  bool runOnModule(Module &M) override { return replaceCBufferAccesses(M); }
  StringRef getPassName() const override { return "DXIL CBuffer Access"; }
  DXILCBufferAccessLegacy() : ModulePass(ID) {}

  static char ID; // Pass identification.
};
char DXILCBufferAccessLegacy::ID = 0;
} // end anonymous namespace

INITIALIZE_PASS(DXILCBufferAccessLegacy, DEBUG_TYPE, "DXIL CBuffer Access",
                false, false)

ModulePass *llvm::createDXILCBufferAccessLegacyPass() {
  return new DXILCBufferAccessLegacy();
}
