//==-LTOInternalize.cpp - LLVM Link Time Optimizer Internalization Utility -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a helper to run the internalization part of LTO.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/LTO/UpdateCompilerUsed.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace llvm;

namespace {

// Helper class that collects AsmUsed and user supplied libcalls.
class PreserveLibCallsAndAsmUsed {
public:
  PreserveLibCallsAndAsmUsed(const StringSet<> &AsmUndefinedRefs,
                             const TargetMachine &TM,
                             SmallPtrSetImpl<const GlobalValue *> &LLVMUsed)
      : AsmUndefinedRefs(AsmUndefinedRefs), TM(TM), LLVMUsed(LLVMUsed) {}

  void findInModule(const Module &TheModule) {
    initializeLibCalls(TheModule);
    for (const Function &F : TheModule)
      findLibCallsAndAsm(F);
    for (const GlobalVariable &GV : TheModule.globals())
      findLibCallsAndAsm(GV);
    for (const GlobalAlias &GA : TheModule.aliases())
      findLibCallsAndAsm(GA);
  }

private:
  // Inputs
  const StringSet<> &AsmUndefinedRefs;
  const TargetMachine &TM;

  // Temps
  llvm::Mangler Mangler;
  StringSet<> Libcalls;

  // Output
  SmallPtrSetImpl<const GlobalValue *> &LLVMUsed;

  // Collect names of runtime library functions. User-defined functions with the
  // same names are added to llvm.compiler.used to prevent them from being
  // deleted by optimizations.
  void initializeLibCalls(const Module &TheModule) {
    TargetLibraryInfoImpl TLII(Triple(TM.getTargetTriple()));
    TargetLibraryInfo TLI(TLII);

    // TargetLibraryInfo has info on C runtime library calls on the current
    // target.
    for (unsigned I = 0, E = static_cast<unsigned>(LibFunc::NumLibFuncs);
         I != E; ++I) {
      LibFunc::Func F = static_cast<LibFunc::Func>(I);
      if (TLI.has(F))
        Libcalls.insert(TLI.getName(F));
    }

    SmallPtrSet<const TargetLowering *, 1> TLSet;

    for (const Function &F : TheModule) {
      const TargetLowering *Lowering =
          TM.getSubtargetImpl(F)->getTargetLowering();

      if (Lowering && TLSet.insert(Lowering).second)
        // TargetLowering has info on library calls that CodeGen expects to be
        // available, both from the C runtime and compiler-rt.
        for (unsigned I = 0, E = static_cast<unsigned>(RTLIB::UNKNOWN_LIBCALL);
             I != E; ++I)
          if (const char *Name =
                  Lowering->getLibcallName(static_cast<RTLIB::Libcall>(I)))
            Libcalls.insert(Name);
    }
  }

  void findLibCallsAndAsm(const GlobalValue &GV) {
    // There are no restrictions to apply to declarations.
    if (GV.isDeclaration())
      return;

    // There is nothing more restrictive than private linkage.
    if (GV.hasPrivateLinkage())
      return;

    // Conservatively append user-supplied runtime library functions to
    // llvm.compiler.used.  These could be internalized and deleted by
    // optimizations like -globalopt, causing problems when later optimizations
    // add new library calls (e.g., llvm.memset => memset and printf => puts).
    // Leave it to the linker to remove any dead code (e.g. with -dead_strip).
    if (isa<Function>(GV) && Libcalls.count(GV.getName()))
      LLVMUsed.insert(&GV);

    SmallString<64> Buffer;
    TM.getNameWithPrefix(Buffer, &GV, Mangler);
    if (AsmUndefinedRefs.count(Buffer))
      LLVMUsed.insert(&GV);
  }
};

} // namespace anonymous

void llvm::updateCompilerUsed(Module &TheModule, const TargetMachine &TM,
                              const StringSet<> &AsmUndefinedRefs) {
  SmallPtrSet<const GlobalValue *, 8> UsedValues;
  PreserveLibCallsAndAsmUsed(AsmUndefinedRefs, TM, UsedValues)
      .findInModule(TheModule);

  if (UsedValues.empty())
    return;

  llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(TheModule.getContext());
  std::vector<Constant *> UsedValuesList;
  for (const auto *GV : UsedValues) {
    Constant *c =
        ConstantExpr::getBitCast(const_cast<GlobalValue *>(GV), i8PTy);
    UsedValuesList.push_back(c);
  }

  GlobalVariable *LLVMUsed = TheModule.getGlobalVariable("llvm.compiler.used");
  if (LLVMUsed) {
    ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
    for (auto &V : Inits->operands())
      UsedValuesList.push_back(cast<Constant>(&V));
    LLVMUsed->eraseFromParent();
  }

  llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, UsedValuesList.size());
  LLVMUsed = new llvm::GlobalVariable(
      TheModule, ATy, false, llvm::GlobalValue::AppendingLinkage,
      llvm::ConstantArray::get(ATy, UsedValuesList), "llvm.compiler.used");

  LLVMUsed->setSection("llvm.metadata");
}
