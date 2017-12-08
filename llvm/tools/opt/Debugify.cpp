//===- Debugify.cpp - Attach synthetic debug info to everything -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file This pass attaches synthetic debug info to everything. It can be used
/// to create targeted tests for debug info preservation.
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

namespace {

bool applyDebugifyMetadata(Module &M) {
  // Skip modules with debug info.
  if (M.getNamedMetadata("llvm.dbg.cu")) {
    errs() << "Debugify: Skipping module with debug info\n";
    return false;
  }

  DIBuilder DIB(M);
  LLVMContext &Ctx = M.getContext();

  // Get a DIType which corresponds to Ty.
  DenseMap<uint64_t, DIType *> TypeCache;
  auto getCachedDIType = [&](Type *Ty) -> DIType * {
    uint64_t Size = M.getDataLayout().getTypeAllocSizeInBits(Ty);
    DIType *&DTy = TypeCache[Size];
    if (!DTy) {
      std::string Name = "ty" + utostr(Size);
      DTy = DIB.createBasicType(Name, Size, dwarf::DW_ATE_unsigned);
    }
    return DTy;
  };

  unsigned NextLine = 1;
  unsigned NextVar = 1;
  auto File = DIB.createFile(M.getName(), "/");
  auto CU =
      DIB.createCompileUnit(dwarf::DW_LANG_C, DIB.createFile(M.getName(), "/"),
                            "debugify", /*isOptimized=*/true, "", 0);

  // Visit each instruction.
  for (Function &F : M) {
    if (F.isDeclaration())
      continue;

    auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
    bool IsLocalToUnit = F.hasPrivateLinkage() || F.hasInternalLinkage();
    auto SP =
        DIB.createFunction(CU, F.getName(), F.getName(), File, NextLine, SPType,
                           IsLocalToUnit, F.hasExactDefinition(), NextLine,
                           DINode::FlagZero, /*isOptimized=*/true);
    F.setSubprogram(SP);
    for (BasicBlock &BB : F) {
      // Attach debug locations.
      for (Instruction &I : BB)
        I.setDebugLoc(DILocation::get(Ctx, NextLine++, 1, SP));

      // Attach debug values.
      for (Instruction &I : BB) {
        // Skip void-valued instructions.
        if (I.getType()->isVoidTy())
          continue;

        // Skip the terminator instruction and any just-inserted intrinsics.
        if (isa<TerminatorInst>(&I) || isa<DbgValueInst>(&I))
          break;

        std::string Name = utostr(NextVar++);
        const DILocation *Loc = I.getDebugLoc().get();
        auto LocalVar = DIB.createAutoVariable(SP, Name, File, Loc->getLine(),
                                               getCachedDIType(I.getType()),
                                               /*AlwaysPreserve=*/true);
        DIB.insertDbgValueIntrinsic(&I, LocalVar, DIB.createExpression(), Loc,
                                    BB.getTerminator());
      }
    }
    DIB.finalizeSubprogram(SP);
  }
  DIB.finalize();

  // Track the number of distinct lines and variables.
  NamedMDNode *NMD = M.getOrInsertNamedMetadata("llvm.debugify");
  auto *IntTy = Type::getInt32Ty(Ctx);
  auto addDebugifyOperand = [&](unsigned N) {
    NMD->addOperand(MDNode::get(
        Ctx, ValueAsMetadata::getConstant(ConstantInt::get(IntTy, N))));
  };
  addDebugifyOperand(NextLine - 1); // Original number of lines.
  addDebugifyOperand(NextVar - 1);  // Original number of variables.
  return true;
}

void checkDebugifyMetadata(Module &M) {
  // Skip modules without debugify metadata.
  NamedMDNode *NMD = M.getNamedMetadata("llvm.debugify");
  if (!NMD)
    return;

  auto getDebugifyOperand = [&](unsigned Idx) -> unsigned {
    return mdconst::extract<ConstantInt>(NMD->getOperand(Idx)->getOperand(0))
        ->getZExtValue();
  };
  unsigned OriginalNumLines = getDebugifyOperand(0);
  unsigned OriginalNumVars = getDebugifyOperand(1);
  bool HasErrors = false;

  // Find missing lines.
  BitVector MissingLines{OriginalNumLines, true};
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      if (isa<DbgValueInst>(&I))
        continue;

      auto DL = I.getDebugLoc();
      if (DL) {
        MissingLines.reset(DL.getLine() - 1);
        continue;
      }

      outs() << "ERROR: Instruction with empty DebugLoc -- ";
      I.print(outs());
      outs() << "\n";
      HasErrors = true;
    }
  }
  for (unsigned Idx : MissingLines.set_bits())
    outs() << "WARNING: Missing line " << Idx + 1 << "\n";

  // Find missing variables.
  BitVector MissingVars{OriginalNumVars, true};
  for (Function &F : M) {
    for (Instruction &I : instructions(F)) {
      auto *DVI = dyn_cast<DbgValueInst>(&I);
      if (!DVI)
        continue;

      unsigned Var = ~0U;
      (void)to_integer(DVI->getVariable()->getName(), Var, 10);
      assert(Var <= OriginalNumVars && "Unexpected name for DILocalVariable");
      MissingVars.reset(Var - 1);
    }
  }
  for (unsigned Idx : MissingVars.set_bits())
    outs() << "ERROR: Missing variable " << Idx + 1 << "\n";
  HasErrors |= MissingVars.count() > 0;

  outs() << "CheckDebugify: " << (HasErrors ? "FAIL" : "PASS") << "\n";
}

/// Attach synthetic debug info to everything.
struct DebugifyPass : public ModulePass {
  bool runOnModule(Module &M) override { return applyDebugifyMetadata(M); }

  DebugifyPass() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static char ID; // Pass identification.
};

/// Check debug info inserted by -debugify for completeness.
struct CheckDebugifyPass : public ModulePass {
  bool runOnModule(Module &M) override {
    checkDebugifyMetadata(M);
    return false;
  }

  CheckDebugifyPass() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static char ID; // Pass identification.
};

} // end anonymous namespace

char DebugifyPass::ID = 0;
static RegisterPass<DebugifyPass> X("debugify",
                                    "Attach debug info to everything");

char CheckDebugifyPass::ID = 0;
static RegisterPass<CheckDebugifyPass> Y("check-debugify",
                                         "Check debug info from -debugify");
