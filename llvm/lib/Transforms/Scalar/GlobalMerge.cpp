//===-- GlobalMerge.cpp - Internal globals merging  -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// This pass merges globals with internal linkage into one. This way all the
// globals which were merged into a biggest one can be addressed using offsets
// from the same base pointer (no need for separate base pointer for each of the
// global). Such a transformation can significantly reduce the register pressure
// when many globals are involved.
//
// For example, consider the code which touches several global variables at
// once:
//
// static int foo[N], bar[N], baz[N];
//
// for (i = 0; i < N; ++i) {
//    foo[i] = bar[i] * baz[i];
// }
//
//  On ARM the addresses of 3 arrays should be kept in the registers, thus
//  this code has quite large register pressure (loop body):
//
//  ldr     r1, [r5], #4
//  ldr     r2, [r6], #4
//  mul     r1, r2, r1
//  str     r1, [r0], #4
//
//  Pass converts the code to something like:
//
//  static struct {
//    int foo[N];
//    int bar[N];
//    int baz[N];
//  } merged;
//
//  for (i = 0; i < N; ++i) {
//    merged.foo[i] = merged.bar[i] * merged.baz[i];
//  }
//
//  and in ARM code this becomes:
//
//  ldr     r0, [r5, #40]
//  ldr     r1, [r5, #80]
//  mul     r0, r1, r0
//  str     r0, [r5], #4
//
//  note that we saved 2 registers here almostly "for free".
// ===---------------------------------------------------------------------===//

#define DEBUG_TYPE "global-merge"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Attributes.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Instructions.h"
#include "llvm/Intrinsics.h"
#include "llvm/Module.h"
#include "llvm/Pass.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumMerged      , "Number of globals merged");
namespace {
  class GlobalMerge : public FunctionPass {
    /// TLI - Keep a pointer of a TargetLowering to consult for determining
    /// target type sizes.
    const TargetLowering *TLI;

    bool doMerge(SmallVectorImpl<GlobalVariable*> &Globals,
                 Module &M, bool isConst) const;

  public:
    static char ID;             // Pass identification, replacement for typeid.
    explicit GlobalMerge(const TargetLowering *tli = 0)
      : FunctionPass(ID), TLI(tli) {
      initializeGlobalMergePass(*PassRegistry::getPassRegistry());
    }

    virtual bool doInitialization(Module &M);
    virtual bool runOnFunction(Function &F);

    const char *getPassName() const {
      return "Merge internal globals";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      FunctionPass::getAnalysisUsage(AU);
    }

    struct GlobalCmp {
      const DataLayout *TD;

      GlobalCmp(const DataLayout *td) : TD(td) { }

      bool operator()(const GlobalVariable *GV1, const GlobalVariable *GV2) {
        Type *Ty1 = cast<PointerType>(GV1->getType())->getElementType();
        Type *Ty2 = cast<PointerType>(GV2->getType())->getElementType();

        return (TD->getTypeAllocSize(Ty1) < TD->getTypeAllocSize(Ty2));
      }
    };
  };
} // end anonymous namespace

char GlobalMerge::ID = 0;
INITIALIZE_PASS(GlobalMerge, "global-merge",
                "Global Merge", false, false)


bool GlobalMerge::doMerge(SmallVectorImpl<GlobalVariable*> &Globals,
                             Module &M, bool isConst) const {
  const DataLayout *TD = TLI->getDataLayout();

  // FIXME: Infer the maximum possible offset depending on the actual users
  // (these max offsets are different for the users inside Thumb or ARM
  // functions)
  unsigned MaxOffset = TLI->getMaximalGlobalOffset();

  // FIXME: Find better heuristics
  std::stable_sort(Globals.begin(), Globals.end(), GlobalCmp(TD));

  Type *Int32Ty = Type::getInt32Ty(M.getContext());

  for (size_t i = 0, e = Globals.size(); i != e; ) {
    size_t j = 0;
    uint64_t MergedSize = 0;
    std::vector<Type*> Tys;
    std::vector<Constant*> Inits;
    for (j = i; j != e; ++j) {
      Type *Ty = Globals[j]->getType()->getElementType();
      MergedSize += TD->getTypeAllocSize(Ty);
      if (MergedSize > MaxOffset) {
        break;
      }
      Tys.push_back(Ty);
      Inits.push_back(Globals[j]->getInitializer());
    }

    StructType *MergedTy = StructType::get(M.getContext(), Tys);
    Constant *MergedInit = ConstantStruct::get(MergedTy, Inits);
    GlobalVariable *MergedGV = new GlobalVariable(M, MergedTy, isConst,
                                                  GlobalValue::InternalLinkage,
                                                  MergedInit, "_MergedGlobals");
    for (size_t k = i; k < j; ++k) {
      Constant *Idx[2] = {
        ConstantInt::get(Int32Ty, 0),
        ConstantInt::get(Int32Ty, k-i)
      };
      Constant *GEP = ConstantExpr::getInBoundsGetElementPtr(MergedGV, Idx);
      Globals[k]->replaceAllUsesWith(GEP);
      Globals[k]->eraseFromParent();
      NumMerged++;
    }
    i = j;
  }

  return true;
}


bool GlobalMerge::doInitialization(Module &M) {
  SmallVector<GlobalVariable*, 16> Globals, ConstGlobals, BSSGlobals;
  const DataLayout *TD = TLI->getDataLayout();
  unsigned MaxOffset = TLI->getMaximalGlobalOffset();
  bool Changed = false;

  // Grab all non-const globals.
  for (Module::global_iterator I = M.global_begin(),
         E = M.global_end(); I != E; ++I) {
    // Merge is safe for "normal" internal globals only
    if (!I->hasLocalLinkage() || I->isThreadLocal() || I->hasSection())
      continue;

    // Ignore fancy-aligned globals for now.
    unsigned Alignment = TD->getPreferredAlignment(I);
    Type *Ty = I->getType()->getElementType();
    if (Alignment > TD->getABITypeAlignment(Ty))
      continue;

    // Ignore all 'special' globals.
    if (I->getName().startswith("llvm.") ||
        I->getName().startswith(".llvm."))
      continue;

    if (TD->getTypeAllocSize(Ty) < MaxOffset) {
      if (TargetLoweringObjectFile::getKindForGlobal(I, TLI->getTargetMachine())
          .isBSSLocal())
        BSSGlobals.push_back(I);
      else if (I->isConstant())
        ConstGlobals.push_back(I);
      else
        Globals.push_back(I);
    }
  }

  if (Globals.size() > 1)
    Changed |= doMerge(Globals, M, false);
  if (BSSGlobals.size() > 1)
    Changed |= doMerge(BSSGlobals, M, false);

  // FIXME: This currently breaks the EH processing due to way how the
  // typeinfo detection works. We might want to detect the TIs and ignore
  // them in the future.
  // if (ConstGlobals.size() > 1)
  //  Changed |= doMerge(ConstGlobals, M, true);

  return Changed;
}

bool GlobalMerge::runOnFunction(Function &F) {
  return false;
}

Pass *llvm::createGlobalMergePass(const TargetLowering *tli) {
  return new GlobalMerge(tli);
}
