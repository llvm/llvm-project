//===-- PISAKernelByValArgsLowering.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PISAKernelByValArgsLowering.h"

#include "PISA.h"
#include "PISADefines.h"

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/PISAAddrSpace.h>

#define DEBUG_TYPE "pisa-kernel-byval-args-lowering"
#define DEBUG_NAME "PISA kernel byval args lowering (Legacy)"

using namespace llvm;
using namespace llvm::PISA;
using namespace llvm::PISAAS;

namespace {
class PISAKernelByValArgsLowering {
public:
  explicit PISAKernelByValArgsLowering(Function &F)
      : F(F), Ctx(F.getContext()) {
    assert(F.getCallingConv() == CallingConv::PISA_KERNEL &&
           "Expected PISA_KERNEL calling convention");
    assert(F.getReturnType()->isVoidTy() && "Expected void return type");
  }

  bool run();

private:
  static constexpr unsigned ByRefAddressSpace =
      static_cast<unsigned>(AddressSpace::CONSTANT);

  AttributeList getNewAttributes(const AttributeList &Attrs) const;
  void transferArgumentUses(Function &NewF);

  Function &F;
  LLVMContext &Ctx;
};

bool PISAKernelByValArgsLowering::run() {
  if (none_of(F.args(), [](Argument &Arg) { return Arg.hasByValAttr(); }))
    return false;

  LLVM_DEBUG(dbgs() << "Lowering kernel aggregate arguments for " << F.getName()
                    << "\n");

  AttributeList NewAttrs = getNewAttributes(F.getAttributes());

  SmallVector<Type *> NewArgTypes;
  transform(F.args(), std::back_inserter(NewArgTypes),
            [this](Argument &Arg) -> Type * {
              if (Arg.hasByValAttr())
                return PointerType::get(Ctx, ByRefAddressSpace);
              return Arg.getType();
            });

  FunctionType *NewFTy =
      FunctionType::get(Type::getVoidTy(Ctx), NewArgTypes, false);
  Function *NewF = Function::Create(NewFTy, F.getLinkage());
  NewF->takeName(&F);
  NewF->setAttributes(NewAttrs);
  NewF->setCallingConv(CallingConv::PISA_KERNEL);

  F.getParent()->getFunctionList().insert(F.getIterator(), NewF);

  // Transfer debug info to the new function
  DISubprogram *DISubprog = F.getSubprogram();
  NewF->setSubprogram(DISubprog);
  F.setSubprogram(nullptr);

  // Splice the body of the old function into the new function.
  NewF->splice(NewF->begin(), &F);

  transferArgumentUses(*NewF);

  F.replaceAllUsesWith(NewF);

  NewF->copyMetadata(&F, 0);
  return true;
}

AttributeList PISAKernelByValArgsLowering::getNewAttributes(
    const AttributeList &Attrs) const {
  AttributeList NewAttrs;

  // Copy function attributes.
  if (AttributeSet FunctionAttrs = Attrs.getFnAttrs();
      FunctionAttrs.hasAttributes()) {
    AttrBuilder AB(Ctx, FunctionAttrs);
    NewAttrs = NewAttrs.addFnAttributes(Ctx, AB);
  }

  for (const Argument &Arg : F.args()) {
    const unsigned Index = Arg.getArgNo();
    AttributeSet ArgAttrs = Attrs.getParamAttrs(Index);
    AttrBuilder AB(Ctx, ArgAttrs);

    if (ArgAttrs.hasAttribute(Attribute::ByVal)) {
      // Replace byval with byref
      assert(cast<PointerType>(Arg.getType())->getAddressSpace() ==
                 static_cast<unsigned>(AddressSpace::PRIVATE) &&
             "Expected private address space");

      Type *ByValTy = ArgAttrs.getByValType();

      AB.removeAttribute(Attribute::ByVal).addByRefAttr(ByValTy);
      NewAttrs = NewAttrs.addParamAttributes(Ctx, Index, AB);
    } else {
      // Copy the argument as is.
      NewAttrs = NewAttrs.addParamAttributes(Ctx, Index, AB);
    }
  }

  return NewAttrs;
}

void PISAKernelByValArgsLowering::transferArgumentUses(Function &NewF) {
  IRBuilder<> Builder(Ctx);

  for (unsigned I = 0, E = F.arg_size(); I != E; ++I) {
    Argument &OldArg = *F.getArg(I);
    Argument &NewArg = *NewF.getArg(I);
    NewArg.takeName(&OldArg);

    if (!OldArg.hasByValAttr()) {
      OldArg.replaceAllUsesWith(&NewArg);
      continue;
    }

    LLVM_DEBUG(dbgs() << "Convert byval argument <" << OldArg << "> to byref <"
                      << NewArg << ">\n");

    SmallVector<Value *, 8> Stack;
    DenseMap<Value *, Value *> Map;
    SmallVector<Instruction *, 8> ToDelete;
    Stack.push_back(&OldArg);
    Map[&OldArg] = &NewArg;

    // Traverse all users of the argument and create GEPs and loads with the
    // new address space. A GEP/memop chain can later be combined into a single
    // ld.param PISA instruction.
    while (!Stack.empty()) {
      Value *V = Stack.back();
      Stack.pop_back();
      for (User *U : V->users()) {
        if (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(U)) {
          assert(GEP->getPointerOperand() == V);
          Stack.push_back(GEP);
          Builder.SetInsertPoint(GEP->getIterator());
          Value *NewGEP = Builder.CreateGEP(
              GEP->getSourceElementType(), Map[V],
              SmallVector<Value *, 4>(GEP->indices()),
              GEP->getName() + ".byref", GEP->getNoWrapFlags());
          LLVM_DEBUG(dbgs() << "Created getelementptr: " << *NewGEP << "\n");
          Map[GEP] = NewGEP;
        } else if (LoadInst *Load = dyn_cast<LoadInst>(U)) {
          Builder.SetInsertPoint(Load->getIterator());
          LoadInst *NewLoad = Builder.CreateAlignedLoad(
              Load->getType(), Map[Load->getPointerOperand()], Load->getAlign(),
              Load->isVolatile(), Load->getName() + ".byref");
          LLVM_DEBUG(dbgs() << "Created load: " << *NewLoad << "\n");
          Load->replaceAllUsesWith(NewLoad);
          ToDelete.push_back(Load);
        } else if (MemCpyInst *MemCpy = dyn_cast<MemCpyInst>(U)) {
          if (MemCpy && MemCpy->getSource() == V) {
            Builder.SetInsertPoint(MemCpy);
            CallInst *NewMemCpy = Builder.CreateMemCpy(
                MemCpy->getDest(), MemCpy->getDestAlign(),
                Map[MemCpy->getSource()], MemCpy->getSourceAlign(),
                MemCpy->getLength(), MemCpy->isVolatile());
            LLVM_DEBUG(dbgs() << "Created memcpy: " << *NewMemCpy << "\n");
            (void)NewMemCpy;
            ToDelete.push_back(MemCpy);
          } else {
            LLVM_DEBUG(dbgs() << "Only memcpy using byval argument as a source "
                                 "can be combined into a ld.param: "
                              << *MemCpy << "\n");
          }
        } else {
          LLVM_DEBUG(
              dbgs()
              << "Byval argument's user can't be combined into a ld.param: "
              << *U << "\n");
        }
      }
    }

    // Perform cleanup: erase replaced memory operations and the chain of GEPs
    // between them and the argument.
    for (Instruction *Inst : ToDelete) {
      Value *Ptr = nullptr;
      if (LoadInst *Load = dyn_cast<LoadInst>(Inst)) {
        Ptr = Load->getPointerOperand();
      } else if (MemCpyInst *MemCpy = dyn_cast<MemCpyInst>(Inst)) {
        Ptr = MemCpy->getSource();
      }
      assert(Ptr);
      Inst->eraseFromParent();
      while (GetElementPtrInst *GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
        Ptr = GEP->getPointerOperand();
        if (GEP->getNumUses() > 0)
          break;
        GEP->eraseFromParent();
      }
    }

    // If all of the argument's users were processed earlier, nothing more needs
    // to be done here.
    if (OldArg.getNumUses() == 0)
      continue;

    // Otherwise, create a copy in the private address space and replace any
    // remaining users.
    Builder.SetInsertPoint(NewF.getEntryBlock().getFirstInsertionPt());
    unsigned AS = static_cast<unsigned>(AddressSpace::PRIVATE);
    Type *ByValTy = OldArg.getAttribute(Attribute::ByVal).getValueAsType();
    AllocaInst *Alloca = Builder.CreateAlloca(
        ByValTy, AS, nullptr, OldArg.getName() + ".private");
    LLVM_DEBUG(dbgs() << "Created alloca: " << *Alloca << "\n");
    const DataLayout &DL = F.getParent()->getDataLayout();
    CallInst *MemCpy = Builder.CreateMemCpy(
        Alloca, Alloca->getAlign(), &NewArg, NewArg.getPointerAlignment(DL),
        DL.getTypeStoreSize(ByValTy));
    LLVM_DEBUG(dbgs() << "Created memcpy: " << *MemCpy << "\n");
    (void)MemCpy;
    OldArg.replaceAllUsesWith(Alloca);
  }
}

class PISAKernelByValArgsLoweringLegacy : public ModulePass {
public:
  static char ID;

  PISAKernelByValArgsLoweringLegacy() : ModulePass(ID) {}

  StringRef getPassName() const override { return DEBUG_NAME; }

  bool runOnModule(Module &M) override {
    bool Changed = false;
    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE;) {
      Function &F = *FI++;
      if (F.isDeclaration() || F.getCallingConv() != CallingConv::PISA_KERNEL)
        continue;
      PISAKernelByValArgsLowering KBVAL(F);
      if (KBVAL.run()) {
        F.eraseFromParent();
        Changed = true;
      }
    }
    return Changed;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    ModulePass::getAnalysisUsage(AU);
  }
};
} // namespace

char PISAKernelByValArgsLoweringLegacy::ID = 0;
INITIALIZE_PASS(PISAKernelByValArgsLoweringLegacy, DEBUG_TYPE, DEBUG_NAME,
                false, false)

ModulePass *llvm::createPISAKernelByValArgsLoweringLegacyPass() {
  return new PISAKernelByValArgsLoweringLegacy();
}

PreservedAnalyses KernelByValArgsLoweringPass::run(Module &M,
                                                   ModuleAnalysisManager &) {
  SmallVector<Function *> ToErase;

  for (Function &F : M) {
    if (F.isDeclaration() || F.getCallingConv() != CallingConv::PISA_KERNEL)
      continue;

    if (PISAKernelByValArgsLowering LKA(F); LKA.run())
      ToErase.push_back(&F);
  }

  if (ToErase.empty())
    return PreservedAnalyses::all();

  for (Function *F : ToErase)
    F->eraseFromParent();

  return PreservedAnalyses::none();
}
