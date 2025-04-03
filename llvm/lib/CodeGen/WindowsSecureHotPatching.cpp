//===------ WindowsHotPatch.cpp - Support for Windows hotpatching ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Provides support for the Windows "Secure Hot-Patching" feature.
//
// Windows contains technology, called "Secure Hot-Patching" (SHP), for securely
// applying hot-patches to a running system. Hot-patches may be applied to the
// kernel, kernel-mode components, device drivers, user-mode system services,
// etc.
//
// SHP relies on integration between many tools, including compiler, linker,
// hot-patch generation tools, and the Windows kernel. This file implements that
// part of the workflow needed in compilers / code generators.
//
// SHP is not intended for productivity scenarios such as Edit-and-Continue or
// interactive development. SHP is intended to minimize downtime during
// installation of Windows OS patches.
//
// In order to work with SHP, LLVM must do all of the following:
//
// * On some architectures (X86, AMD64), the function prolog must begin with
//   hot-patchable instructions. This is handled by the MSVC `/hotpatch` option
//   and the equivalent `-fms-hotpatch` function. This is necessary because we
//   generally cannot anticipate which functions will need to be patched in the
//   future. This option ensures that a function can be hot-patched in the
//   future, but does not actually generate any hot-patch for it.
//
// * For a selected set of functions that are being hot-patched (which are
//   identified using command-line options), LLVM must generate the
//   `S_HOTPATCHFUNC` CodeView record (symbol). This record indicates that a
//   function was compiled with hot-patching enabled.
//
//   This implementation uses the `MarkedForWindowsHotPatching` attribute to
//   annotate those functions that were marked for hot-patching by command-line
//   parameters. The attribute may be specified by a language front-end by
//   setting an attribute when a function is created in LLVM IR, or it may be
//   set by passing LLVM arguments.
//
// * For those functions that are hot-patched, LLVM must rewrite references to
//   global variables so that they are indirected through a `__ref_*` pointer
//   variable.  For each global variable, that is accessed by a hot-patched
//   function, e.g. `FOO`, a `__ref_FOO` global pointer variable is created and
//   all references to the original `FOO` are rewritten as dereferences of the
//   `__ref_FOO` pointer.
//
//   Some globals do not need `__ref_*` indirection. The pointer indirection
//   behavior can be disabled for these globals by marking them with the
//   `AllowDirectAccessInHotPatchFunction`.
//
// References
//
// * "Hotpatching on Windows":
//   https://techcommunity.microsoft.com/blog/windowsosplatform/hotpatching-on-windows/2959541
//
// * "Hotpatch for Windows client now available":
//   https://techcommunity.microsoft.com/blog/windows-itpro-blog/hotpatch-for-windows-client-now-available/4399808
//
// * "Get hotpatching for Windows Server":
//   https://www.microsoft.com/en-us/windows-server/blog/2025/04/24/tired-of-all-the-restarts-get-hotpatching-for-windows-server/
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

#define DEBUG_TYPE "windows-secure-hot-patch"

// A file containing list of mangled function names to mark for hot patching.
static cl::opt<std::string> LLVMMSSecureHotPatchFunctionsFile(
    "ms-secure-hotpatch-functions-file", cl::value_desc("filename"),
    cl::desc("A file containing list of mangled function names to mark for "
             "Windows Secure Hot-Patching"));

// A list of mangled function names to mark for hot patching.
static cl::list<std::string> LLVMMSSecureHotPatchFunctionsList(
    "ms-secure-hotpatch-functions-list", cl::value_desc("list"),
    cl::desc("A list of mangled function names to mark for Windows Secure "
             "Hot-Patching"),
    cl::CommaSeparated);

namespace {

class WindowsSecureHotPatching : public ModulePass {
  struct GlobalVariableUse {
    GlobalVariable *GV;
    Instruction *User;
    unsigned Op;
  };

public:
  static char ID;

  WindowsSecureHotPatching() : ModulePass(ID) {
    initializeWindowsSecureHotPatchingPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

  bool doInitialization(Module &) override;
  bool runOnModule(Module &M) override { return false; }

private:
  bool
  runOnFunction(Function &F,
                SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping);
};

} // end anonymous namespace

char WindowsSecureHotPatching::ID = 0;

INITIALIZE_PASS(WindowsSecureHotPatching, "windows-secure-hot-patch",
                "Mark functions for Windows hot patch support", false, false)
ModulePass *llvm::createWindowsSecureHotPatchingPass() {
  return new WindowsSecureHotPatching();
}

// Find functions marked with Attribute::MarkedForWindowsHotPatching and modify
// their code (if necessary) to account for accesses to global variables.
//
// This runs during doInitialization() instead of runOnModule() because it needs
// to run before CodeViewDebug::collectGlobalVariableInfo().
bool WindowsSecureHotPatching::doInitialization(Module &M) {
  // The front end may have already marked functions for hot-patching. However,
  // we also allow marking functions by passing -ms-hotpatch-functions-file or
  // -ms-hotpatch-functions-list directly to LLVM. This allows hot-patching to
  // work with languages that have not yet updated their front-ends.
  if (!LLVMMSSecureHotPatchFunctionsFile.empty() ||
      !LLVMMSSecureHotPatchFunctionsList.empty()) {
    std::vector<std::string> HotPatchFunctionsList;

    if (!LLVMMSSecureHotPatchFunctionsFile.empty()) {
      auto BufOrErr =
          llvm::MemoryBuffer::getFile(LLVMMSSecureHotPatchFunctionsFile);
      if (BufOrErr) {
        const llvm::MemoryBuffer &FileBuffer = **BufOrErr;
        for (llvm::line_iterator I(FileBuffer.getMemBufferRef(), true), E;
             I != E; ++I)
          HotPatchFunctionsList.push_back(std::string{*I});
      } else {
        M.getContext().diagnose(llvm::DiagnosticInfoGeneric{
            llvm::Twine("failed to open hotpatch functions file "
                        "(--ms-hotpatch-functions-file): ") +
            LLVMMSSecureHotPatchFunctionsFile + llvm::Twine(" : ") +
            BufOrErr.getError().message()});
      }
    }

    if (!LLVMMSSecureHotPatchFunctionsList.empty())
      for (const auto &FuncName : LLVMMSSecureHotPatchFunctionsList)
        HotPatchFunctionsList.push_back(FuncName);

    // Build a set for quick lookups. This points into HotPatchFunctionsList, so
    // HotPatchFunctionsList must live longer than HotPatchFunctionsSet.
    llvm::SmallSet<llvm::StringRef, 16> HotPatchFunctionsSet;
    for (const auto &FuncName : HotPatchFunctionsList)
      HotPatchFunctionsSet.insert(llvm::StringRef{FuncName});

    // Iterate through all of the functions and check whether they need to be
    // marked for hotpatching using the list provided directly to LLVM.
    for (auto &F : M.functions()) {
      // Ignore declarations that are not definitions.
      if (F.isDeclarationForLinker())
        continue;

      if (HotPatchFunctionsSet.contains(F.getName()))
        F.addFnAttr(Attribute::MarkedForWindowsHotPatching);
    }
  }

  SmallDenseMap<GlobalVariable *, GlobalVariable *> RefMapping;
  bool MadeChanges = false;
  for (auto &F : M.functions()) {
    if (F.hasFnAttribute(Attribute::MarkedForWindowsHotPatching)) {
      if (runOnFunction(F, RefMapping))
        MadeChanges = true;
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
// indirections go through a new global variable, named `__ref_<Foo>` where
// `<Foo>` is the original symbol name of the global variable.
//
// This function handles rewriting accesses to global variables, but the
// generation of S_HOTPATCHFUNC occurs in
// CodeViewDebug::emitHotPatchInformation().
//
// Returns true if any changes were made to the function.
bool WindowsSecureHotPatching::runOnFunction(
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

  if (GVUses.empty())
    return false;

  const llvm::DISubprogram *Subprogram = F.getSubprogram();
  llvm::DICompileUnit *Unit =
      Subprogram != nullptr ? Subprogram->getUnit() : nullptr;
  llvm::DIFile *File = Subprogram != nullptr ? Subprogram->getFile() : nullptr;
  DIBuilder DebugInfo{*F.getParent(), true, Unit};

  for (auto &GVUse : GVUses) {
    IRBuilder<> Builder(GVUse.User);

    // Get or create a new global variable that points to the old one and whose
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
      DataLayout Layout = F.getParent()->getDataLayout();
      DIType *DebugType = DebugInfo.createPointerType(
          nullptr, Layout.getTypeSizeInBits(GVUse.GV->getValueType()));
      DIGlobalVariableExpression *GVE =
          DebugInfo.createGlobalVariableExpression(
              Unit, ReplaceWithRefGV->getName(), StringRef{}, File,
              /*LineNo*/ 0, DebugType,
              /*IsLocalToUnit*/ false);
      ReplaceWithRefGV->addDebugInfo(GVE);
    }

    // Now replace the use of that global variable with the new one (via a load
    // since it is a pointer to the old global variable).
    LoadInst *LoadedRefGV =
        Builder.CreateLoad(ReplaceWithRefGV->getValueType(), ReplaceWithRefGV);
    GVUse.User->setOperand(GVUse.Op, LoadedRefGV);
  }

  if (Subprogram != nullptr)
    DebugInfo.finalize();

  return true;
}
