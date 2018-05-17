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

bool isFunctionSkipped(Function &F) {
  return F.isDeclaration() || !F.hasExactDefinition();
}

bool applyDebugifyMetadata(Module &M,
                           iterator_range<Module::iterator> Functions,
                           StringRef Banner) {
  // Skip modules with debug info.
  if (M.getNamedMetadata("llvm.dbg.cu")) {
    errs() << Banner << "Skipping module with debug info\n";
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
  auto CU = DIB.createCompileUnit(dwarf::DW_LANG_C, File,
                            "debugify", /*isOptimized=*/true, "", 0);

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
  assert(NMD->getNumOperands() == 2 &&
         "llvm.debugify should have exactly 2 operands!");
  return true;
}

bool checkDebugifyMetadata(Module &M,
                           iterator_range<Module::iterator> Functions,
                           StringRef NameOfWrappedPass,
                           StringRef Banner,
                           bool Strip) {
  // Skip modules without debugify metadata.
  NamedMDNode *NMD = M.getNamedMetadata("llvm.debugify");
  if (!NMD) {
    errs() << Banner << "Skipping module without debugify metadata\n";
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

      errs() << "ERROR: Instruction with empty DebugLoc in function ";
      errs() << F.getName() << " --";
      I.print(errs());
      errs() << "\n";
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
    errs() << "WARNING: Missing line " << Idx + 1 << "\n";

  for (unsigned Idx : MissingVars.set_bits())
    errs() << "ERROR: Missing variable " << Idx + 1 << "\n";
  HasErrors |= MissingVars.count() > 0;

  errs() << Banner << " [" << NameOfWrappedPass << "]: "
         << (HasErrors ? "FAIL" : "PASS") << '\n';
  if (HasErrors) {
    errs() << "Module IR Dump\n";
    M.print(errs(), nullptr, false);
  }

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
                                 NameOfWrappedPass, "CheckFunctionDebugify", Strip);
  }

  CheckDebugifyFunctionPass(bool Strip = false, StringRef NameOfWrappedPass = "")
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

ModulePass *createDebugifyModulePass() {
  return new DebugifyModulePass();
}

FunctionPass *createDebugifyFunctionPass() {
  return new DebugifyFunctionPass();
}

PreservedAnalyses NewPMDebugifyPass::run(Module &M, ModuleAnalysisManager &) {
  applyDebugifyMetadata(M, M.functions(), "ModuleDebugify: ");
  return PreservedAnalyses::all();
}

ModulePass *createCheckDebugifyModulePass(bool Strip, StringRef NameOfWrappedPass) {
  return new CheckDebugifyModulePass(Strip, NameOfWrappedPass);
}

FunctionPass *createCheckDebugifyFunctionPass(bool Strip, StringRef NameOfWrappedPass) {
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
static RegisterPass<CheckDebugifyModulePass> CDM("check-debugify",
                                         "Check debug info from -debugify");

char DebugifyFunctionPass::ID = 0;
static RegisterPass<DebugifyFunctionPass> DF("debugify-function",
                                    "Attach debug info to a function");

char CheckDebugifyFunctionPass::ID = 0;
static RegisterPass<CheckDebugifyFunctionPass> CDF("check-debugify-function",
                                         "Check debug info from -debugify-function");
