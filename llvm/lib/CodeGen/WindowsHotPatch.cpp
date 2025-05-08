//===------ WindowsHotPatch.cpp - Support for Windows hotpatching ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Marks functions with the `marked_for_windows_hot_patching` attribute.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;

#define DEBUG_TYPE "windows-hot-patch"

// A file containing list of mangled function names to mark for hot patching.
static cl::opt<std::string> LLVMMSHotPatchFunctionsFile(
    "ms-hotpatch-functions-file", cl::value_desc("filename"),
    cl::desc("A file containing list of mangled function names to mark for hot "
             "patching"));

// A list of mangled function names to mark for hot patching.
static cl::list<std::string> LLVMMSHotPatchFunctionsList(
    "ms-hotpatch-functions-list", cl::value_desc("list"),
    cl::desc("A list of mangled function names to mark for hot patching"),
    cl::CommaSeparated);

namespace {

class WindowsHotPatch : public ModulePass {
  struct GlobalVariableUse {
    GlobalVariable *GV;
    Instruction *User;
    unsigned Op;
  };

public:
  static char ID;

  WindowsHotPatch() : ModulePass(ID) {
    initializeWindowsHotPatchPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool runOnModule(Module &M) override;

private:
  bool
  runOnFunction(Function &F,
                SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping);
  void replaceGlobalVariableUses(
      Function &F, SmallVectorImpl<GlobalVariableUse> &GVUses,
      SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping,
      DIBuilder &DebugInfo);
};

} // end anonymous namespace

char WindowsHotPatch::ID = 0;

INITIALIZE_PASS(WindowsHotPatch, "windows-hot-patch",
                "Mark functions for Windows hot patch support", false, false)
ModulePass *llvm::createWindowsHotPatch() { return new WindowsHotPatch(); }

// Find functions marked with Attribute::MarkedForWindowsHotPatching and modify
// their code (if necessary) to account for accesses to global variables.
bool WindowsHotPatch::runOnModule(Module &M) {
  // If the target OS is not Windows, then check that there are no functions
  // marked for Windows hot-patching.
  if (!M.getTargetTriple().isOSBinFormatCOFF()) {
    if (!LLVMMSHotPatchFunctionsFile.empty()) {
      M.getContext().diagnose(llvm::DiagnosticInfoGeneric{
          llvm::Twine("--ms-hotpatch-functions-file is only supported when "
                      "target OS is Windows")});
    }

    if (!LLVMMSHotPatchFunctionsList.empty()) {
      M.getContext().diagnose(llvm::DiagnosticInfoGeneric{
          llvm::Twine("--ms-hotpatch-functions-list is only supported when "
                      "target OS is Windows")});
    }

    // Functions may have already been marked for hot-patching by the front-end.
    // Check for any functions marked for hot-patching and report an error if
    // any are found.
    for (auto &F : M.functions()) {
      if (F.hasFnAttribute(Attribute::MarkedForWindowsHotPatching)) {
        M.getContext().diagnose(llvm::DiagnosticInfoGeneric{
            llvm::Twine("function is marked for Windows hot-patching, but the "
                        "target OS is not Windows: ") +
            F.getName()});
      }
    }
    return false;
  }

  // The front end may have already marked functions for hot-patching. However,
  // we also allow marking functions by passing -ms-hotpatch-functions-file or
  // -ms-hotpatch-functions-list directly to LLVM. This allows hot-patching to
  // work with languages that have not yet updated their front-ends.
  if (!LLVMMSHotPatchFunctionsFile.empty() ||
      !LLVMMSHotPatchFunctionsList.empty()) {
    std::vector<std::string> HotPatchFunctionsList;

    if (!LLVMMSHotPatchFunctionsFile.empty()) {
      auto BufOrErr = llvm::MemoryBuffer::getFile(LLVMMSHotPatchFunctionsFile);
      if (BufOrErr) {
        const llvm::MemoryBuffer &FileBuffer = **BufOrErr;
        for (llvm::line_iterator I(FileBuffer.getMemBufferRef(), true), E;
             I != E; ++I) {
          auto Line = llvm::StringRef(*I).trim();
          if (!Line.empty()) {
            HotPatchFunctionsList.push_back(std::string{Line});
          }
        }
      } else {
        M.getContext().diagnose(llvm::DiagnosticInfoGeneric{
            llvm::Twine("failed to open hotpatch functions file "
                        "(--ms-hotpatch-functions-file): ") +
            LLVMMSHotPatchFunctionsFile + llvm::Twine(" : ") +
            BufOrErr.getError().message()});
      }
    }

    if (!LLVMMSHotPatchFunctionsList.empty()) {
      for (const auto &FuncName : LLVMMSHotPatchFunctionsList) {
        HotPatchFunctionsList.push_back(FuncName);
      }
    }

    // Build a set for quick lookups. This points into HotPatchFunctionsList, so
    // HotPatchFunctionsList must live longer than HotPatchFunctionsSet.
    llvm::SmallSet<llvm::StringRef, 16> HotPatchFunctionsSet;
    for (const auto &FuncName : HotPatchFunctionsList) {
      HotPatchFunctionsSet.insert(llvm::StringRef{FuncName});
    }

    // Iterate through all of the functions and check whether they need to be
    // marked for hotpatching using the list provided directly to LLVM.
    for (auto &F : M.functions()) {
      // Ignore declarations that are not definitions.
      if (F.isDeclarationForLinker()) {
        continue;
      }

      if (HotPatchFunctionsSet.contains(F.getName())) {
        F.addFnAttr(Attribute::MarkedForWindowsHotPatching);
      }
    }
  }

  SmallDenseMap<GlobalVariable *, GlobalVariable *> RefMapping;
  bool MadeChanges = false;
  for (auto &F : M.functions()) {
    if (F.hasFnAttribute(Attribute::MarkedForWindowsHotPatching)) {
      if (runOnFunction(F, RefMapping)) {
        MadeChanges = true;
      }
    }
  }
  return MadeChanges;
}

// Processes a function that is marked for hot-patching.
//
// If a function is marked for hot-patching, we generate an S_HOTPATCHFUNC
// CodeView debug symbol. Tools that generate hot-patches look for
// S_HOTPATCHFUNC in final PDBs so that they can find functions that have been
// hot-patched and so that they can distinguish hot-patched functions from
// non-hot-patched functions.
//
// Also, in functions that are hot-patched, we must indirect all access to
// (mutable) global variables through a pointer. This pointer may point into the
// unpatched ("base") binary or may point into the patched image, depending on
// whether a hot-patch was loaded as a patch or as a base image.  These
// indirections go through a new global variable, `named __ref_<Foo>` where
// `<Foo>` is the original symbol name of the global variable.
//
// This function handles rewriting accesses to global variables, but the
// generation of S_HOTPATCHFUNC occurs in
// CodeViewDebug::emitHotPatchInformation().
//
// Returns true if any changes were made to the function.
bool WindowsHotPatch::runOnFunction(
    Function &F,
    SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping) {
  SmallVector<GlobalVariableUse, 32> GVUses;
  for (auto &I : instructions(F)) {
    for (auto &U : I.operands()) {
      // Discover all uses of GlobalVariable, these will need to be replaced.
      GlobalVariable *GV = dyn_cast<GlobalVariable>(&U);
      if ((GV != nullptr) &&
          !GV->hasAttribute(Attribute::AllowDirectAccessInHotPatchFunction)) {
        unsigned OpNo = &U - I.op_begin();
        GVUses.push_back({GV, &I, OpNo});
      }
    }
  }

  if (!GVUses.empty()) {
    const llvm::DISubprogram *Subprogram = F.getSubprogram();
    DIBuilder DebugInfo{*F.getParent(), true,
                        Subprogram != nullptr ? Subprogram->getUnit()
                                              : nullptr};
    replaceGlobalVariableUses(F, GVUses, RefMapping, DebugInfo);
    if (Subprogram != nullptr) {
      DebugInfo.finalize();
    }
    return true;
  } else {
    return false;
  }
}

void WindowsHotPatch::replaceGlobalVariableUses(
    Function &F, SmallVectorImpl<GlobalVariableUse> &GVUses,
    SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping,
    DIBuilder &DebugInfo) {
  for (auto &GVUse : GVUses) {
    IRBuilder<> Builder(GVUse.User);

    // Get or create a new global variable that points to the old one and who's
    // name begins with `__ref_`.
    GlobalVariable *&ReplaceWithRefGV =
        RefMapping.try_emplace(GVUse.GV).first->second;
    if (ReplaceWithRefGV == nullptr) {
      Constant *AddrOfOldGV = ConstantExpr::getGetElementPtr(
          Builder.getPtrTy(), GVUse.GV, ArrayRef<Value *>{});
      ReplaceWithRefGV =
          new GlobalVariable(*F.getParent(), Builder.getPtrTy(), true,
                             GlobalValue::InternalLinkage, AddrOfOldGV,
                             Twine("__ref_").concat(GVUse.GV->getName()),
                             nullptr, GlobalVariable::NotThreadLocal);

      // Create debug info for the replacement global variable.
      DISubprogram *SP = F.getSubprogram();
      DataLayout Layout = F.getParent()->getDataLayout();
      DIType *DebugType = DebugInfo.createPointerType(
          nullptr, Layout.getTypeSizeInBits(GVUse.GV->getValueType()));
      DIGlobalVariableExpression *GVE =
          DebugInfo.createGlobalVariableExpression(
              SP != nullptr ? SP->getUnit() : nullptr,
              ReplaceWithRefGV->getName(), StringRef{},
              SP != nullptr ? SP->getFile() : nullptr, /*LineNo*/ 0, DebugType,
              /*IsLocalToUnit*/ false);
      ReplaceWithRefGV->addDebugInfo(GVE);
    }

    // Now replace the use of that global variable with the new one (via a load
    // since it is a pointer to the old global variable).
    LoadInst *LoadedRefGV =
        Builder.CreateLoad(ReplaceWithRefGV->getValueType(), ReplaceWithRefGV);
    GVUse.User->setOperand(GVUse.Op, LoadedRefGV);
  }
}
