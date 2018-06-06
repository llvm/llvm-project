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

#include "PassPrinters.h"
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

cl::opt<bool> Quiet("debugify-quiet",
                    cl::desc("Suppress verbose debugify output"));

raw_ostream &dbg() { return Quiet ? nulls() : errs(); }

bool isFunctionSkipped(Function &F) {
  return F.isDeclaration() || !F.hasExactDefinition();
}

/// Find the basic block's terminating instruction.
///
/// Special care is needed to handle musttail and deopt calls, as these behave
/// like (but are in fact not) terminators.
Instruction *findTerminatingInstruction(BasicBlock &BB) {
  if (auto *I = BB.getTerminatingMustTailCall())
    return I;
  if (auto *I = BB.getTerminatingDeoptimizeCall())
    return I;
  return BB.getTerminator();
}

bool applyDebugifyMetadata(Module &M,
                           iterator_range<Module::iterator> Functions,
                           StringRef Banner) {
  // Skip modules with debug info.
  if (M.getNamedMetadata("llvm.dbg.cu")) {
    dbg() << Banner << "Skipping module with debug info\n";
    return false;
  }

  DIBuilder DIB(M);
  LLVMContext &Ctx = M.getContext();

  // Get a DIType which corresponds to Ty.
  DenseMap<uint64_t, DIType *> TypeCache;
  auto getCachedDIType = [&](Type *Ty) -> DIType * {
    uint64_t Size =
        Ty->isSized() ? M.getDataLayout().getTypeAllocSizeInBits(Ty) : 0;
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
  auto CU = DIB.createCompileUnit(dwarf::DW_LANG_C, File, "debugify",
                                  /*isOptimized=*/true, "", 0);

  // Visit each instruction.
  for (Function &F : Functions) {
    if (isFunctionSkipped(F))
      continue;

    auto SPType = DIB.createSubroutineType(DIB.getOrCreateTypeArray(None));
    bool IsLocalToUnit = F.hasPrivateLinkage() || F.hasInternalLinkage();
    auto SP =
        DIB.createFunction(CU, F.getName(), F.getName(), File, NextLine, SPType,
                           IsLocalToUnit, /*isDefinition=*/true, NextLine,
                           DINode::FlagZero, /*isOptimized=*/true);
    F.setSubprogram(SP);
    for (BasicBlock &BB : F) {
      // Attach debug locations.
      for (Instruction &I : BB)
        I.setDebugLoc(DILocation::get(Ctx, NextLine++, 1, SP));

      // Inserting debug values into EH pads can break IR invariants.
      if (BB.isEHPad())
        continue;

      // Find the terminating instruction, after which no debug values are
      // attached.
      Instruction *LastInst = findTerminatingInstruction(BB);
      assert(LastInst && "Expected basic block with a terminator");

      // Maintain an insertion point which can't be invalidated when updates
      // are made.
      BasicBlock::iterator InsertPt = BB.getFirstInsertionPt();
      assert(InsertPt != BB.end() && "Expected to find an insertion point");
      Instruction *InsertBefore = &*InsertPt;

      // Attach debug values.
      for (Instruction *I = &*BB.begin(); I != LastInst; I = I->getNextNode()) {
        // Skip void-valued instructions.
        if (I->getType()->isVoidTy())
          continue;

        // Phis and EH pads must be grouped at the beginning of the block.
        // Only advance the insertion point when we finish visiting these.
        if (!isa<PHINode>(I) && !I->isEHPad())
          InsertBefore = I->getNextNode();

        std::string Name = utostr(NextVar++);
        const DILocation *Loc = I->getDebugLoc().get();
        auto LocalVar = DIB.createAutoVariable(SP, Name, File, Loc->getLine(),
                                               getCachedDIType(I->getType()),
                                               /*AlwaysPreserve=*/true);
        DIB.insertDbgValueIntrinsic(I, LocalVar, DIB.createExpression(), Loc,
                                    InsertBefore);
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
  assert(NMD->getNumOperands() == 2 &&
         "llvm.debugify should have exactly 2 operands!");

  // Claim that this synthetic debug info is valid.
  StringRef DIVersionKey = "Debug Info Version";
  if (!M.getModuleFlag(DIVersionKey))
    M.addModuleFlag(Module::Warning, DIVersionKey, DEBUG_METADATA_VERSION);

  return true;
}

bool checkDebugifyMetadata(Module &M,
                           iterator_range<Module::iterator> Functions,
                           StringRef NameOfWrappedPass, StringRef Banner,
                           bool Strip) {
  // Skip modules without debugify metadata.
  NamedMDNode *NMD = M.getNamedMetadata("llvm.debugify");
  if (!NMD) {
    dbg() << Banner << "Skipping module without debugify metadata\n";
    return false;
  }

  auto getDebugifyOperand = [&](unsigned Idx) -> unsigned {
    return mdconst::extract<ConstantInt>(NMD->getOperand(Idx)->getOperand(0))
        ->getZExtValue();
  };
  assert(NMD->getNumOperands() == 2 &&
         "llvm.debugify should have exactly 2 operands!");
  unsigned OriginalNumLines = getDebugifyOperand(0);
  unsigned OriginalNumVars = getDebugifyOperand(1);
  bool HasErrors = false;

  BitVector MissingLines{OriginalNumLines, true};
  BitVector MissingVars{OriginalNumVars, true};
  for (Function &F : Functions) {
    if (isFunctionSkipped(F))
      continue;

    // Find missing lines.
    for (Instruction &I : instructions(F)) {
      if (isa<DbgValueInst>(&I))
        continue;

      auto DL = I.getDebugLoc();
      if (DL && DL.getLine() != 0) {
        MissingLines.reset(DL.getLine() - 1);
        continue;
      }

      dbg() << "ERROR: Instruction with empty DebugLoc in function ";
      dbg() << F.getName() << " --";
      I.print(dbg());
      dbg() << "\n";
      HasErrors = true;
    }

    // Find missing variables.
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

  // Print the results.
  for (unsigned Idx : MissingLines.set_bits())
    dbg() << "WARNING: Missing line " << Idx + 1 << "\n";

  for (unsigned Idx : MissingVars.set_bits())
    dbg() << "ERROR: Missing variable " << Idx + 1 << "\n";
  HasErrors |= MissingVars.count() > 0;

  dbg() << Banner;
  if (!NameOfWrappedPass.empty())
    dbg() << " [" << NameOfWrappedPass << "]";
  dbg() << ": " << (HasErrors ? "FAIL" : "PASS") << '\n';

  // Strip the Debugify Metadata if required.
  if (Strip) {
    StripDebugInfo(M);
    M.eraseNamedMetadata(NMD);
    return true;
  }

  return false;
}

/// ModulePass for attaching synthetic debug info to everything, used with the
/// legacy module pass manager.
struct DebugifyModulePass : public ModulePass {
  bool runOnModule(Module &M) override {
    return applyDebugifyMetadata(M, M.functions(), "ModuleDebugify: ");
  }

  DebugifyModulePass() : ModulePass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static char ID; // Pass identification.
};

/// FunctionPass for attaching synthetic debug info to instructions within a
/// single function, used with the legacy module pass manager.
struct DebugifyFunctionPass : public FunctionPass {
  bool runOnFunction(Function &F) override {
    Module &M = *F.getParent();
    auto FuncIt = F.getIterator();
    return applyDebugifyMetadata(M, make_range(FuncIt, std::next(FuncIt)),
                                 "FunctionDebugify: ");
  }

  DebugifyFunctionPass() : FunctionPass(ID) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static char ID; // Pass identification.
};

/// ModulePass for checking debug info inserted by -debugify, used with the
/// legacy module pass manager.
struct CheckDebugifyModulePass : public ModulePass {
  bool runOnModule(Module &M) override {
    return checkDebugifyMetadata(M, M.functions(), NameOfWrappedPass,
                                 "CheckModuleDebugify", Strip);
  }

  CheckDebugifyModulePass(bool Strip = false, StringRef NameOfWrappedPass = "")
      : ModulePass(ID), Strip(Strip), NameOfWrappedPass(NameOfWrappedPass) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static char ID; // Pass identification.

private:
  bool Strip;
  StringRef NameOfWrappedPass;
};

/// FunctionPass for checking debug info inserted by -debugify-function, used
/// with the legacy module pass manager.
struct CheckDebugifyFunctionPass : public FunctionPass {
  bool runOnFunction(Function &F) override {
    Module &M = *F.getParent();
    auto FuncIt = F.getIterator();
    return checkDebugifyMetadata(M, make_range(FuncIt, std::next(FuncIt)),
                                 NameOfWrappedPass, "CheckFunctionDebugify",
                                 Strip);
  }

  CheckDebugifyFunctionPass(bool Strip = false,
                            StringRef NameOfWrappedPass = "")
      : FunctionPass(ID), Strip(Strip), NameOfWrappedPass(NameOfWrappedPass) {}

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
  }

  static char ID; // Pass identification.

private:
  bool Strip;
  StringRef NameOfWrappedPass;
};

} // end anonymous namespace

ModulePass *createDebugifyModulePass() { return new DebugifyModulePass(); }

FunctionPass *createDebugifyFunctionPass() {
  return new DebugifyFunctionPass();
}

PreservedAnalyses NewPMDebugifyPass::run(Module &M, ModuleAnalysisManager &) {
  applyDebugifyMetadata(M, M.functions(), "ModuleDebugify: ");
  return PreservedAnalyses::all();
}

ModulePass *createCheckDebugifyModulePass(bool Strip,
                                          StringRef NameOfWrappedPass) {
  return new CheckDebugifyModulePass(Strip, NameOfWrappedPass);
}

FunctionPass *createCheckDebugifyFunctionPass(bool Strip,
                                              StringRef NameOfWrappedPass) {
  return new CheckDebugifyFunctionPass(Strip, NameOfWrappedPass);
}

PreservedAnalyses NewPMCheckDebugifyPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  checkDebugifyMetadata(M, M.functions(), "", "CheckModuleDebugify", false);
  return PreservedAnalyses::all();
}

char DebugifyModulePass::ID = 0;
static RegisterPass<DebugifyModulePass> DM("debugify",
                                           "Attach debug info to everything");

char CheckDebugifyModulePass::ID = 0;
static RegisterPass<CheckDebugifyModulePass>
    CDM("check-debugify", "Check debug info from -debugify");

char DebugifyFunctionPass::ID = 0;
static RegisterPass<DebugifyFunctionPass> DF("debugify-function",
                                             "Attach debug info to a function");

char CheckDebugifyFunctionPass::ID = 0;
static RegisterPass<CheckDebugifyFunctionPass>
    CDF("check-debugify-function", "Check debug info from -debugify-function");
