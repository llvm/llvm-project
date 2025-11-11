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
// Rewriting references to global variables has some complexity.
//
// For ordinary instructions that reference GlobalVariables, we rewrite the
// operand of the instruction to a Load of the __ref_* variable.
//
// For constant expressions, we have to convert the constant expression (and
// transitively all constant expressions in its parent chain) to non-constant
// expressions, i.e. to a sequence of instructions.
//
// Pass 1:
//   * Enumerate all instructions in all basic blocks.
//
//   * If an instruction references a GlobalVariable (and it is not marked
//     as being ignored), then we create (if necessary) the __ref_* variable
//     for the GlobalVariable reference. However, we do not yet modify the
//     Instruction.
//
//   * If an instruction has an operand that is a ConstantExpr and the
//     ConstantExpression tree contains a reference to a GlobalVariable, then
//     we similarly create __ref_*. Similarly, we do not yet modify the
//     Instruction or the ConstantExpr tree.
//
// After Pass 1 completes, we will know whether we found any references to
// globals in this pass.  If the function does not use any globals (and most
// functions do not use any globals), then we return immediately.
//
// If a function does reference globals, then we iterate the list of globals
// used by this function and we generate Load instructions for each (unique)
// global.
//
// Next, we do another pass over all instructions:
//
// Pass 2:
//   * Re-visit the instructions that were found in Pass 1.
//
//   * If an instruction operand is a GlobalVariable, then look up the
//   replacement
//     __ref_* global variable and the Value that came from the Load instruction
//     for it.  Replace the operand of the GlobalVariable with the Load Value.
//
//   * If an instruction operand is a ConstantExpr, then recursively examine the
//     operands of all instructions in the ConstantExpr tree.  If an operand is
//     a GlobalVariable, then replace the operand with the result of the load
//     *and* convert the ConstantExpr to a non-constant instruction.  This
//     instruction will need to be inserted into the BB of the instruction whose
//     operand is being modified, ideally immediately before the instruction
//     being modified.
//
// Limitations
//
// This feature is not intended to work in every situation. There are many
// legitimate code changes (patches) for which it is not possible to generate
// a hot-patch. Developers who are writing hot-patches are expected to
// understand the limitations.
//
// Tools which generate hot-patch metadata may also check that certain
// variables are upheld, and some of these invariants may be global (may require
// whole-program knowledge, not available in any single compiland). However,
// such tools are not required to be perfect; they are also best-effort.
//
// For these reasons, the hot-patching support implemented in this file is
// "best effort". It does not recognize every possible code pattern that could
// be patched, nor does it generate diagnostics for certain code patterns that
// could result in a binary that does not work with hot-patching. For example,
// const GlobalVariables that point to other non-const GlobalVariables are not
// compatible with hot-patching because they cannot use __ref_*-based
// redirection.
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

struct GlobalVariableUse {
  // GlobalVariable *GV;
  Instruction *User;
  unsigned Op;
};

class WindowsSecureHotPatching : public ModulePass {
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
      auto BufOrErr = MemoryBuffer::getFile(LLVMMSSecureHotPatchFunctionsFile);
      if (BufOrErr) {
        const MemoryBuffer &FileBuffer = **BufOrErr;
        for (line_iterator I(FileBuffer.getMemBufferRef(), true), E; I != E;
             ++I)
          HotPatchFunctionsList.push_back(std::string{*I});
      } else {
        M.getContext().diagnose(DiagnosticInfoGeneric{
            Twine("failed to open hotpatch functions file "
                  "(--ms-hotpatch-functions-file): ") +
            LLVMMSSecureHotPatchFunctionsFile + Twine(" : ") +
            BufOrErr.getError().message()});
      }
    }

    if (!LLVMMSSecureHotPatchFunctionsList.empty())
      for (const auto &FuncName : LLVMMSSecureHotPatchFunctionsList)
        HotPatchFunctionsList.push_back(FuncName);

    // Build a set for quick lookups. This points into HotPatchFunctionsList, so
    // HotPatchFunctionsList must live longer than HotPatchFunctionsSet.
    SmallSet<StringRef, 16> HotPatchFunctionsSet;
    for (const auto &FuncName : HotPatchFunctionsList)
      HotPatchFunctionsSet.insert(StringRef{FuncName});

    // Iterate through all of the functions and check whether they need to be
    // marked for hotpatching using the list provided directly to LLVM.
    for (auto &F : M.functions()) {
      // Ignore declarations that are not definitions.
      if (F.isDeclarationForLinker())
        continue;

      if (HotPatchFunctionsSet.contains(F.getName()))
        F.addFnAttr("marked_for_windows_hot_patching");
    }
  }

  SmallDenseMap<GlobalVariable *, GlobalVariable *> RefMapping;
  bool MadeChanges = false;
  for (auto &F : M.functions()) {
    if (F.hasFnAttribute("marked_for_windows_hot_patching")) {
      if (runOnFunction(F, RefMapping))
        MadeChanges = true;
    }
  }
  return MadeChanges;
}

static bool TypeContainsPointers(Type *ty) {
  switch (ty->getTypeID()) {
  case Type::PointerTyID:
    return true;

  case Type::ArrayTyID:
    return TypeContainsPointers(ty->getArrayElementType());

  case Type::StructTyID: {
    unsigned NumElements = ty->getStructNumElements();
    for (unsigned I = 0; I < NumElements; ++I) {
      if (TypeContainsPointers(ty->getStructElementType(I))) {
        return true;
      }
    }
    return false;
  }

  default:
    return false;
  }
}

// Returns true if GV needs redirection through a __ref_* variable.
static bool globalVariableNeedsRedirect(GlobalVariable *GV) {
  // If a global variable is explictly marked as allowing access in hot-patched
  // functions, then do not redirect it.
  if (GV->hasAttribute("allow_direct_access_in_hot_patch_function"))
    return false;

  // If the global variable is not a constant, then we want to redirect it.
  if (!GV->isConstant()) {
    if (GV->getName().starts_with("??_R")) {
      // This is the name mangling prefix that MSVC uses for RTTI data.
      // Clang is currently generating RTTI data that is marked non-constant.
      // We override that and treat it like it is constant.
      return false;
    }

    // In general, if a global variable is not a constant, then redirect it.
    return true;
  }

  // If the type of GV cannot contain pointers, then it cannot point to
  // other global variables. In this case, there is no need for redirects.
  // For example, string literals do not contain pointers.
  return TypeContainsPointers(GV->getValueType());
}

// Get or create a new global variable that points to the old one and whose
// name begins with `__ref_`.
//
// In hot-patched images, the __ref_* variables point to global variables in
// the original (unpatched) image. Hot-patched functions in the hot-patch
// image use these __ref_* variables to access global variables. This ensures
// that all code (both unpatched and patched) is using the same instances of
// global variables.
//
// The Windows hot-patch infrastructure handles modifying these __ref_*
// variables. By default, they are initialized with pointers to the equivalent
// global variables, so when a hot-patch module is loaded *as* a base image
// (such as after a system reboot), hot-patch functions will access the
// instances of global variables that are compiled into the hot-patch image.
// This is the desired outcome, since in this situation (normal boot) the
// hot-patch image *is* the base image.
//
// When we create the GlobalVariable for the __ref_* variable, we must create
// it as a *non-constant* global variable. The __ref_* pointers will not change
// during the runtime of the program, so it is tempting to think that they
// should be constant. However, they still need to be updateable by the
// hot-patching infrastructure. Also, if the GlobalVariable is created as a
// constant, then the LLVM optimizer will assume that it can dereference the
// definition of the __ref_* variable at compile time, which defeats the
// purpose of the indirection (pointer).
//
// The RefMapping table spans the entire module, not just a single function.
static GlobalVariable *getOrCreateRefVariable(
    Function &F, SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping,
    GlobalVariable *GV) {
  GlobalVariable *&ReplaceWithRefGV = RefMapping.try_emplace(GV).first->second;
  if (ReplaceWithRefGV != nullptr) {
    // We have already created a __ref_* pointer for this GlobalVariable.
    return ReplaceWithRefGV;
  }

  Module *M = F.getParent();

  const DISubprogram *Subprogram = F.getSubprogram();
  DICompileUnit *Unit = Subprogram != nullptr ? Subprogram->getUnit() : nullptr;
  DIFile *File = Subprogram != nullptr ? Subprogram->getFile() : nullptr;
  DIBuilder DebugInfo{*F.getParent(), true, Unit};

  auto PtrTy = PointerType::get(M->getContext(), 0);

  Constant *AddrOfOldGV =
      ConstantExpr::getGetElementPtr(PtrTy, GV, ArrayRef<Value *>{});

  GlobalVariable *RefGV =
      new GlobalVariable(*M, PtrTy, false, GlobalValue::LinkOnceAnyLinkage,
                         AddrOfOldGV, Twine("__ref_").concat(GV->getName()),
                         nullptr, GlobalVariable::NotThreadLocal);

  // RefGV is created with isConstant = false, but we want to place RefGV into
  // .rdata, not .data.  It is important that the GlobalVariable be mutable
  // from the compiler's point of view, so that the optimizer does not remove
  // the global variable entirely and replace all references to it with its
  // initial value.
  //
  // When the Windows hot-patch loader applies a hot-patch, it maps the
  // pages of .rdata as read/write so that it can set each __ref_* variable
  // to point to the original variable in the base image. Afterward, pages in
  // .rdata are remapped as read-only. This protects the __ref_* variables from
  // being overwritten during execution.
  RefGV->setSection(".rdata");

  // Create debug info for the replacement global variable.
  DataLayout Layout = M->getDataLayout();
  DIType *DebugType = DebugInfo.createPointerType(
      nullptr, Layout.getTypeSizeInBits(GV->getValueType()));
  DIGlobalVariableExpression *GVE = DebugInfo.createGlobalVariableExpression(
      Unit, RefGV->getName(), StringRef{}, File,
      /*LineNo*/ 0, DebugType,
      /*IsLocalToUnit*/ false);
  RefGV->addDebugInfo(GVE);

  // Store the __ref_* in RefMapping so that future calls use the same RefGV.
  ReplaceWithRefGV = RefGV;

  return RefGV;
}

// Given a ConstantExpr, this searches for GlobalVariable references within
// the expression tree.  If found, it will generate instructions and will
// return a non-null Value* that points to the new root instruction.
//
// If C does not contain any GlobalVariable references, this returns nullptr.
//
// If this function creates new instructions, then it will insert them
// before InsertionPoint.
static Value *rewriteGlobalVariablesInConstant(
    Constant *C, SmallDenseMap<GlobalVariable *, Value *> &GVLoadMap,
    IRBuilder<> &IRBuilderAtEntry) {
  if (C->getValueID() == Value::GlobalVariableVal) {
    GlobalVariable *GV = cast<GlobalVariable>(C);
    if (globalVariableNeedsRedirect(GV)) {
      return GVLoadMap.at(GV);
    } else {
      return nullptr;
    }
  }

  // Scan the operands of this expression.

  SmallVector<Value *, 8> ReplacedValues;
  bool ReplacedAnyOperands = false;

  unsigned NumOperands = C->getNumOperands();
  for (unsigned OpIndex = 0; OpIndex < NumOperands; ++OpIndex) {
    Value *OldValue = C->getOperand(OpIndex);
    Value *ReplacedValue = nullptr;
    if (Constant *OldConstant = dyn_cast<Constant>(OldValue)) {
      ReplacedValue = rewriteGlobalVariablesInConstant(OldConstant, GVLoadMap,
                                                       IRBuilderAtEntry);
    }
    // Do not use short-circuiting, here. We need to traverse the whole tree.
    ReplacedAnyOperands |= ReplacedValue != nullptr;
    ReplacedValues.push_back(ReplacedValue);
  }

  // If none of our operands were replaced, then don't rewrite this expression.
  if (!ReplacedAnyOperands) {
    return nullptr;
  }

  // We need to rewrite this expression. Convert this constant expression
  // to an instruction, then replace any operands as needed.
  Instruction *NewInst = cast<ConstantExpr>(C)->getAsInstruction();
  for (unsigned OpIndex = 0; OpIndex < NumOperands; ++OpIndex) {
    Value *ReplacedValue = ReplacedValues[OpIndex];
    if (ReplacedValue != nullptr) {
      NewInst->setOperand(OpIndex, ReplacedValue);
    }
  }

  // Insert the new instruction before the reference instruction.
  IRBuilderAtEntry.Insert(NewInst);

  return NewInst;
}

static bool searchConstantExprForGlobalVariables(
    Value *V, SmallDenseMap<GlobalVariable *, Value *> &GVLoadMap,
    SmallVector<GlobalVariableUse> &GVUses) {

  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V)) {
    if (globalVariableNeedsRedirect(GV)) {
      GVLoadMap[GV] = nullptr;
      return true;
    } else {
      return false;
    }
  }

  if (User *U = dyn_cast<User>(V)) {
    unsigned NumOperands = U->getNumOperands();
    bool FoundAny = false;
    for (unsigned OpIndex = 0; OpIndex < NumOperands; ++OpIndex) {
      Value *Op = U->getOperand(OpIndex);
      // Do not use short-circuiting, here. We need to traverse the whole tree.
      FoundAny |= searchConstantExprForGlobalVariables(Op, GVLoadMap, GVUses);
    }
    return FoundAny;
  } else {
    return false;
  }
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
// Returns true if any global variable references were found and rewritten.
bool WindowsSecureHotPatching::runOnFunction(
    Function &F,
    SmallDenseMap<GlobalVariable *, GlobalVariable *> &RefMapping) {
  // Scan the function for references to global variables. If we find such a
  // reference, create (if necessary) the __ref_* variable, then add an entry
  // to the GVUses table.
  //
  // We ignore references to global variables if the variable is marked with
  // AllowDirectAccessInHotPatchFunction.

  SmallDenseMap<GlobalVariable *, Value *> GVLoadMap;
  SmallVector<GlobalVariableUse> GVUses;

  for (auto &I : instructions(F)) {
    unsigned NumOperands = I.getNumOperands();
    for (unsigned OpIndex = 0; OpIndex < NumOperands; ++OpIndex) {
      Value *V = I.getOperand(OpIndex);

      bool FoundAnyGVUses = false;

      switch (V->getValueID()) {
      case Value::GlobalVariableVal: {
        // Discover all uses of GlobalVariable, these will need to be replaced.
        GlobalVariable *GV = cast<GlobalVariable>(V);
        if (globalVariableNeedsRedirect(GV)) {
          GVLoadMap.insert(std::make_pair(GV, nullptr));
          FoundAnyGVUses = true;
        }
        break;
      }

      case Value::ConstantExprVal: {
        ConstantExpr *CE = cast<ConstantExpr>(V);
        if (searchConstantExprForGlobalVariables(CE, GVLoadMap, GVUses)) {
          FoundAnyGVUses = true;
        }
        break;
      }

      default:
        break;
      }

      if (FoundAnyGVUses) {
        GVUses.push_back(GlobalVariableUse{&I, OpIndex});
      }
    }
  }

  // If this function did not reference any global variables then we have no
  // work to do. Most functions do not access global variables.
  if (GVUses.empty()) {
    return false;
  }

  // We know that there is at least one instruction that needs to be rewritten.
  // Generate a Load instruction for each unique GlobalVariable used by this
  // function. The Load instructions are inserted at the beginning of the
  // entry block. Since entry blocks cannot contain PHI instructions, there is
  // no need to skip PHI instructions.

  // We use a single IRBuilder for inserting Load instructions as well as the
  // constants that we convert to instructions. Because constants do not
  // depend on any dynamic values (they're constant, after all!), it is safe
  // to move them to the start of entry BB.

  auto &EntryBlock = F.getEntryBlock();
  IRBuilder<> IRBuilderAtEntry(&EntryBlock, EntryBlock.begin());

  for (auto &[GV, LoadValue] : GVLoadMap) {
    assert(LoadValue == nullptr);
    GlobalVariable *RefGV = getOrCreateRefVariable(F, RefMapping, GV);
    LoadValue = IRBuilderAtEntry.CreateLoad(RefGV->getValueType(), RefGV);
  }

  const DISubprogram *Subprogram = F.getSubprogram();
  DICompileUnit *Unit = Subprogram != nullptr ? Subprogram->getUnit() : nullptr;
  DIBuilder DebugInfo{*F.getParent(), true, Unit};

  // Go back to the instructions and rewrite their uses of GlobalVariable.
  // Because a ConstantExpr can be a tree, it may reference more than one
  // GlobalVariable.

  for (auto &GVUse : GVUses) {
    Value *OldOperandValue = GVUse.User->getOperand(GVUse.Op);
    Value *NewOperandValue;

    switch (OldOperandValue->getValueID()) {
    case Value::GlobalVariableVal: {
      // This is easy. Look up the replacement value and store the operand.
      Value *OperandValue = GVUse.User->getOperand(GVUse.Op);
      GlobalVariable *GV = cast<GlobalVariable>(OperandValue);
      NewOperandValue = GVLoadMap.at(GV);
      break;
    }

    case Value::ConstantExprVal: {
      // Walk the recursive tree of the ConstantExpr. If we find a
      // GlobalVariable then replace it with the loaded value and rewrite
      // the ConstantExpr to an Instruction and insert it before the
      // current instruction.
      Value *OperandValue = GVUse.User->getOperand(GVUse.Op);
      ConstantExpr *CE = cast<ConstantExpr>(OperandValue);
      NewOperandValue =
          rewriteGlobalVariablesInConstant(CE, GVLoadMap, IRBuilderAtEntry);
      assert(NewOperandValue != nullptr);
      break;
    }

    default:
      // We should only ever get here because a GVUse was created in the first
      // pass, and this only happens for GlobalVariableVal and ConstantExprVal.
      llvm_unreachable_internal(
          "unexpected Value in second pass of hot-patching");
      break;
    }

    GVUse.User->setOperand(GVUse.Op, NewOperandValue);
  }

  return true;
}
