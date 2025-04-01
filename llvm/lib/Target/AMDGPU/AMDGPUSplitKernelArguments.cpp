//===--- AMDGPUSplitKernelArguments.cpp - Split kernel arguments ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// \file This pass flats struct-type kernel arguments. It eliminates unused
// fields and only keeps used fields. The objective is to facilitate preloading
// of kernel arguments by later passes.
//
//===----------------------------------------------------------------------===//
#include "AMDGPU.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Transforms/Utils/Cloning.h"

#define DEBUG_TYPE "amdgpu-split-kernel-arguments"

using namespace llvm;

namespace {
static cl::opt<bool> EnableSplitKernelArgs(
    "amdgpu-enable-split-kernel-args",
    cl::desc("Enable splitting of AMDGPU kernel arguments"), cl::init(false));

class AMDGPUSplitKernelArguments : public ModulePass {
public:
  static char ID;

  AMDGPUSplitKernelArguments() : ModulePass(ID) {}

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

private:
  bool processFunction(Function &F);
};

} // end anonymous namespace

bool AMDGPUSplitKernelArguments::processFunction(Function &F) {
  const DataLayout &DL = F.getParent()->getDataLayout();
  if (F.isDeclaration())
    return false;

  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty())
    return false;

  SmallVector<std::tuple<unsigned, unsigned, uint64_t>, 8> NewArgMappings;
  DenseMap<Argument *, SmallVector<LoadInst *, 8>> ArgToLoadsMap;
  DenseMap<Argument *, SmallVector<GetElementPtrInst *, 8>> ArgToGEPsMap;
  SmallVector<Argument *, 8> StructArgs;
  SmallVector<Type *, 8> NewArgTypes;

  auto convertAddressSpace = [](Type *Ty) -> Type * {
    if (auto *PtrTy = dyn_cast<PointerType>(Ty)) {
      if (PtrTy->getAddressSpace() == AMDGPUAS::FLAT_ADDRESS) {
        return PointerType::get(PtrTy->getContext(), AMDGPUAS::GLOBAL_ADDRESS);
      }
    }
    return Ty;
  };

  // Collect struct arguments and new argument types
  unsigned OriginalArgIndex = 0;
  unsigned NewArgIndex = 0;
  for (Argument &Arg : F.args()) {
    if (Arg.use_empty()) {
      NewArgTypes.push_back(convertAddressSpace(Arg.getType()));
      NewArgMappings.push_back(
          std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    PointerType *PT = dyn_cast<PointerType>(Arg.getType());
    if (!PT) {
      NewArgTypes.push_back(Arg.getType());
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    const bool IsByRef = Arg.hasByRefAttr();
    if (!IsByRef) {
      NewArgTypes.push_back(Arg.getType());
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    Type *ArgTy = Arg.getParamByRefType();
    StructType *ST = dyn_cast<StructType>(ArgTy);
    if (!ST) {
      NewArgTypes.push_back(Arg.getType());
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
      ++OriginalArgIndex;
      continue;
    }

    bool AllLoadsOrGEPs = true;
    SmallVector<LoadInst *, 8> Loads;
    SmallVector<GetElementPtrInst *, 8> GEPs;
    for (User *U : Arg.users()) {
      LLVM_DEBUG(dbgs() << "  User: " << *U << "\n");
      if (auto *LI = dyn_cast<LoadInst>(U)) {
        Loads.push_back(LI);
      } else if (auto *GEP = dyn_cast<GetElementPtrInst>(U)) {
        GEPs.push_back(GEP);
        for (User *GEPUser : GEP->users()) {
          LLVM_DEBUG(dbgs() << "    GEP User: " << *GEPUser << "\n");
          if (auto *GEPLoad = dyn_cast<LoadInst>(GEPUser)) {
            Loads.push_back(GEPLoad);
          } else {
            AllLoadsOrGEPs = false;
            break;
          }
        }
      } else {
        AllLoadsOrGEPs = false;
        break;
      }
      if (!AllLoadsOrGEPs)
        break;
    }
    LLVM_DEBUG(dbgs() << "  AllLoadsOrGEPs: "
                      << (AllLoadsOrGEPs ? "true" : "false") << "\n");

    if (AllLoadsOrGEPs) {
      StructArgs.push_back(&Arg);
      ArgToLoadsMap[&Arg] = Loads;
      ArgToGEPsMap[&Arg] = GEPs;
      for (LoadInst *LI : Loads) {
        Type *NewType = convertAddressSpace(LI->getType());
        NewArgTypes.push_back(NewType);

        // Compute offset
        uint64_t Offset = 0;
        if (auto *GEP = dyn_cast<GetElementPtrInst>(LI->getPointerOperand())) {
          APInt OffsetAPInt(DL.getPointerSizeInBits(), 0);
          if (GEP->accumulateConstantOffset(DL, OffsetAPInt))
            Offset = OffsetAPInt.getZExtValue();
        }

        // Map each new argument to the original argument index and offset
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, Offset));
        ++NewArgIndex;
      }
    } else {
      NewArgTypes.push_back(convertAddressSpace(Arg.getType()));
      // Include mapping if indices have changed
      if (NewArgIndex != OriginalArgIndex)
        NewArgMappings.push_back(
            std::make_tuple(NewArgIndex, OriginalArgIndex, 0));
      ++NewArgIndex;
    }
    ++OriginalArgIndex;
  }

  if (StructArgs.empty())
    return false;

  // Collect function and return attributes
  AttributeList OldAttrs = F.getAttributes();
  AttributeSet FnAttrs = OldAttrs.getFnAttrs();
  AttributeSet RetAttrs = OldAttrs.getRetAttrs();

  // Create new function type
  FunctionType *NewFT =
      FunctionType::get(F.getReturnType(), NewArgTypes, F.isVarArg());
  Function *NewF =
      Function::Create(NewFT, F.getLinkage(), F.getAddressSpace(), F.getName());
  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);
  NewF->takeName(&F);
  NewF->setVisibility(F.getVisibility());
  if (F.hasComdat())
    NewF->setComdat(F.getComdat());
  NewF->setDSOLocal(F.isDSOLocal());
  NewF->setUnnamedAddr(F.getUnnamedAddr());
  NewF->setCallingConv(F.getCallingConv());

  // Build new parameter attributes
  SmallVector<AttributeSet, 8> NewArgAttrSets;
  NewArgIndex = 0;
  for (Argument &Arg : F.args()) {
    if (ArgToLoadsMap.count(&Arg)) {
      for (LoadInst *LI : ArgToLoadsMap[&Arg]) {
        (void)LI;
        NewArgAttrSets.push_back(AttributeSet());
        ++NewArgIndex;
      }
    } else {
      AttributeSet ArgAttrs = OldAttrs.getParamAttrs(Arg.getArgNo());
      NewArgAttrSets.push_back(ArgAttrs);
      ++NewArgIndex;
    }
  }

  // Build the new AttributeList
  AttributeList NewAttrList =
      AttributeList::get(F.getContext(), FnAttrs, RetAttrs, NewArgAttrSets);
  NewF->setAttributes(NewAttrList);

  // Add the mapping to the old arguments as function argument
  // attribute in the format "OriginalArgIndex:Offset"
  for (const auto &Info : NewArgMappings) {
    unsigned NewArgIdx, OrigArgIdx;
    uint64_t Offset;
    std::tie(NewArgIdx, OrigArgIdx, Offset) = Info;
    NewF->addParamAttr(
        NewArgIdx,
        Attribute::get(NewF->getContext(), "amdgpu-original-arg",
                       (Twine(OrigArgIdx) + ":" + Twine(Offset)).str()));
  }

  LLVM_DEBUG(dbgs() << "New empty function:\n" << *NewF << '\n');

  NewF->splice(NewF->begin(), &F);

  // Map old arguments and loads to new arguments
  DenseMap<Value *, Value *> VMap;
  auto NewArgIt = NewF->arg_begin();
  for (Argument &Arg : F.args()) {
    if (ArgToLoadsMap.count(&Arg)) {
      for (LoadInst *LI : ArgToLoadsMap[&Arg]) {
        NewArgIt->takeName(LI);
        Value *NewArg = &*NewArgIt++;
        if (isa<PointerType>(NewArg->getType()) &&
            isa<PointerType>(LI->getType())) {
          IRBuilder<> Builder(LI);
          Value *CastedArg = Builder.CreatePointerBitCastOrAddrSpaceCast(
              NewArg, LI->getType());
          VMap[LI] = CastedArg;
        } else {
          VMap[LI] = NewArg;
        }
      }
      PoisonValue *PoisonArg = PoisonValue::get(Arg.getType());
      Arg.replaceAllUsesWith(PoisonArg);
    } else {
      NewArgIt->takeName(&Arg);
      Value *NewArg = &*NewArgIt;
      if (isa<PointerType>(NewArg->getType()) &&
          isa<PointerType>(Arg.getType())) {
        IRBuilder<> Builder(&*NewF->begin()->begin());
        Value *CastedArg =
            Builder.CreatePointerBitCastOrAddrSpaceCast(NewArg, Arg.getType());
        Arg.replaceAllUsesWith(CastedArg);
      } else {
        Arg.replaceAllUsesWith(NewArg);
      }
      ++NewArgIt;
    }
  }

  // Replace LoadInsts with new arguments
  for (auto &Entry : ArgToLoadsMap) {
    for (LoadInst *LI : Entry.second) {
      Value *NewArg = VMap[LI];
      LI->replaceAllUsesWith(NewArg);
      LI->eraseFromParent();
    }
  }

  // Erase GEPs
  for (auto &Entry : ArgToGEPsMap) {
    for (GetElementPtrInst *GEP : Entry.second) {
      GEP->replaceAllUsesWith(PoisonValue::get(GEP->getType()));
      GEP->eraseFromParent();
    }
  }

  LLVM_DEBUG(dbgs() << "New function after transformation:\n" << *NewF << '\n');

  F.replaceAllUsesWith(NewF);
  F.eraseFromParent();

  return true;
}

bool AMDGPUSplitKernelArguments::runOnModule(Module &M) {
  if (!EnableSplitKernelArgs)
    return false;
  bool Changed = false;
  SmallVector<Function *, 16> FunctionsToProcess;

  for (Function &F : M) {
    if (F.isDeclaration())
      continue;
    FunctionsToProcess.push_back(&F);
  }

  for (Function *F : FunctionsToProcess) {
    if (F->isDeclaration())
      continue;
    Changed |= processFunction(*F);
  }

  return Changed;
}

INITIALIZE_PASS_BEGIN(AMDGPUSplitKernelArguments, DEBUG_TYPE,
                      "AMDGPU Split Kernel Arguments", false, false)
INITIALIZE_PASS_END(AMDGPUSplitKernelArguments, DEBUG_TYPE,
                    "AMDGPU Split Kernel Arguments", false, false)

char AMDGPUSplitKernelArguments::ID = 0;

ModulePass *llvm::createAMDGPUSplitKernelArgumentsPass() {
  return new AMDGPUSplitKernelArguments();
}

PreservedAnalyses
AMDGPUSplitKernelArgumentsPass::run(Module &M, ModuleAnalysisManager &AM) {
  AMDGPUSplitKernelArguments Splitter;
  bool Changed = Splitter.runOnModule(M);

  if (!Changed)
    return PreservedAnalyses::all();

  return PreservedAnalyses::none();
}
