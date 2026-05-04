//===-- EJitRegisterBitcode.cpp - EmbeddedJIT Bitcode Extraction ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/EmbeddedJIT/EJitPasses.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;
using namespace llvm::ejit;

static bool hasMDStringEntry(const MDNode *Node, StringRef Name) {
  if (!Node)
    return false;
  for (const MDOperand &Op : Node->operands()) {
    auto *Sub = dyn_cast<MDNode>(Op.get());
    if (Sub && Sub->getNumOperands() > 0)
      if (auto *S = dyn_cast<MDString>(Sub->getOperand(0)))
        if (S->getString() == Name)
          return true;
  }
  return false;
}

static void collectEntryFunctions(Module &M,
                                  SmallVectorImpl<Function *> &EntryFuncs) {
  for (Function &F : M.functions()) {
    MDNode *MD = F.getMetadata(MD_EJIT_METADATA);
    if (hasMDStringEntry(MD, TAG_EJIT_ENTRY))
      EntryFuncs.push_back(&F);
  }
}

static void collectReferencedGlobals(Function &F,
                                     SetVector<GlobalVariable *> &Globals) {
  for (BasicBlock &BB : F)
    for (Instruction &I : BB)
      for (Value *Op : I.operands())
        if (auto *GV = dyn_cast<GlobalVariable>(Op->stripPointerCasts()))
          if (!GV->isConstant() || GV->hasMetadata(MD_EJIT_METADATA))
            Globals.insert(GV);
}

static void computeTransitiveClosure(
    const SmallVectorImpl<Function *> &EntryFuncs,
    SetVector<Function *> &ClosureFuncs,
    SetVector<GlobalVariable *> &ClosureGlobals) {

  SmallVector<Function *, 16> Worklist(EntryFuncs.begin(), EntryFuncs.end());
  while (!Worklist.empty()) {
    Function *F = Worklist.pop_back_val();
    if (!ClosureFuncs.insert(F))
      continue;
    collectReferencedGlobals(*F, ClosureGlobals);
    for (BasicBlock &BB : *F)
      for (Instruction &I : BB)
        if (auto *CI = dyn_cast<CallInst>(&I))
          if (Function *Callee = CI->getCalledFunction())
            if (!Callee->isDeclaration() && !Callee->isIntrinsic())
              Worklist.push_back(Callee);
  }
}

static std::string extractAndSerialize(Module &M,
    const SetVector<Function *> &Funcs,
    const SetVector<GlobalVariable *> &Globals) {

  auto Extracted = CloneModule(M);

  DenseSet<StringRef> FuncNames;
  for (Function *F : Funcs)
    FuncNames.insert(F->getName());

  DenseSet<StringRef> GlobalNames;
  for (GlobalVariable *GV : Globals)
    GlobalNames.insert(GV->getName());

  SmallVector<Function *, 16> FuncsToDelete;
  for (Function &F : Extracted->functions())
    if (!FuncNames.count(F.getName()))
      FuncsToDelete.push_back(&F);
  for (Function *F : FuncsToDelete) {
    F->replaceAllUsesWith(UndefValue::get(F->getType()));
    if (!F->isDeclaration())
      F->deleteBody();
    F->eraseFromParent();
  }

  SmallVector<GlobalVariable *, 16> GVToDelete;
  for (GlobalVariable &GV : Extracted->globals())
    if (!GlobalNames.count(GV.getName()))
      GVToDelete.push_back(&GV);
  for (GlobalVariable *GV : GVToDelete) {
    GV->replaceAllUsesWith(UndefValue::get(GV->getType()));
    GV->eraseFromParent();
  }

  std::string Bitcode;
  raw_string_ostream OS(Bitcode);
  WriteBitcodeToFile(*Extracted, OS);
  OS.flush();
  return Bitcode;
}

static GlobalVariable *embedBitcode(Module &M, const std::string &Bitcode) {
  LLVMContext &Ctx = M.getContext();
  SmallVector<uint8_t, 0> Bytes;
  Bytes.reserve(Bitcode.size());
  for (char C : Bitcode)
    Bytes.push_back(static_cast<uint8_t>(C));

  auto *ArrTy = ArrayType::get(Type::getInt8Ty(Ctx), Bitcode.size());
  auto *Const = ConstantDataArray::get(Ctx, Bytes);
  auto *GV = new GlobalVariable(M, ArrTy, true, GlobalValue::InternalLinkage,
                                Const, GV_EJIT_BITCODE);
  GV->setSection(SECT_EJIT_BITCODE);
  GV->setAlignment(Align(1));
  return GV;
}

static void generateRegisterCall(Module &M, GlobalVariable *BitcodeGV,
                                 const SmallVectorImpl<Function *> &EntryFuncs) {
  LLVMContext &Ctx = M.getContext();
  auto *VoidTy = Type::getVoidTy(Ctx);
  auto *PtrTy = PointerType::getUnqual(Ctx);
  auto *I64Ty = Type::getInt64Ty(Ctx);

  M.getOrInsertFunction(FN_REGISTER_BITCODE,
      FunctionType::get(VoidTy, {PtrTy, PtrTy, I64Ty}, false));

  Function *AutoReg = M.getFunction(FN_AUTO_REGISTER);
  if (!AutoReg) {
    AutoReg = Function::Create(FunctionType::get(VoidTy, false),
                               GlobalValue::InternalLinkage,
                               FN_AUTO_REGISTER, &M);
    BasicBlock::Create(Ctx, "entry", AutoReg);
    ReturnInst::Create(Ctx, &AutoReg->getEntryBlock());
  }

  BasicBlock *EntryBB = &AutoReg->getEntryBlock();
  Instruction *Ret = EntryBB->getTerminator();
  FunctionCallee Callee = M.getFunction(FN_REGISTER_BITCODE);

  for (Function *F : EntryFuncs) {
    IRBuilder<> Builder(Ret);
    Builder.CreateCall(Callee, {
        Builder.CreateGlobalString(F->getName()),
        Builder.CreateBitCast(BitcodeGV, PtrTy),
        ConstantInt::get(I64Ty, BitcodeGV->getValueType()->getArrayNumElements())
    });
  }

  // Add to global_ctors
  bool HasEntry = false;
  if (GlobalVariable *Ctors = M.getGlobalVariable(CTORS_GLOBAL))
    if (auto *Arr = dyn_cast<ConstantArray>(Ctors->getInitializer()))
      for (auto &Op : Arr->operands())
        if (auto *CS = dyn_cast<ConstantStruct>(Op))
          if (auto *F = dyn_cast<Function>(CS->getOperand(1)))
            if (F == AutoReg) { HasEntry = true; break; }

  if (!HasEntry) {
    auto *Ty = StructType::get(Ctx, {Type::getInt32Ty(Ctx),
                                      AutoReg->getType(), PtrTy});
    auto *Entry = ConstantStruct::get(Ty,
        {ConstantInt::get(Type::getInt32Ty(Ctx), EJIT_CTOR_PRIORITY), AutoReg,
         ConstantPointerNull::get(PtrTy)});
    if (GlobalVariable *Ctors = M.getGlobalVariable(CTORS_GLOBAL)) {
      if (auto *Arr = dyn_cast<ConstantArray>(Ctors->getInitializer())) {
        SmallVector<Constant *, 8> E;
        for (auto &Op : Arr->operands()) E.push_back(cast<Constant>(Op));
        E.push_back(Entry);
        Ctors->setInitializer(ConstantArray::get(
            ArrayType::get(Ty, E.size()), E));
      }
    } else {
      auto *ATy = ArrayType::get(Ty, 1);
      new GlobalVariable(M, ATy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(ATy, {Entry}), CTORS_GLOBAL);
    }
  }
}

PreservedAnalyses
EJitRegisterBitcodePass::run(Module &M, ModuleAnalysisManager &) {
  SmallVector<Function *, 4> EntryFuncs;
  collectEntryFunctions(M, EntryFuncs);
  if (EntryFuncs.empty())
    return PreservedAnalyses::all();

  SetVector<Function *> ClosureFuncs;
  SetVector<GlobalVariable *> ClosureGlobals;
  computeTransitiveClosure(EntryFuncs, ClosureFuncs, ClosureGlobals);
  if (ClosureFuncs.empty())
    return PreservedAnalyses::all();

  std::string Bitcode = extractAndSerialize(M, ClosureFuncs, ClosureGlobals);
  GlobalVariable *BitcodeGV = embedBitcode(M, Bitcode);
  generateRegisterCall(M, BitcodeGV, EntryFuncs);

  return PreservedAnalyses::none();
}
