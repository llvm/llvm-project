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

#include "LTOInternalize.h"

#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Mangler.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/IPO/Internalize.h"

using namespace llvm;

namespace {
// Helper class that populate the array of symbols used in inlined assembly.
class ComputeAsmUsed {
public:
  ComputeAsmUsed(const StringSet<> &AsmUndefinedRefs, const TargetMachine &TM,
                 const Module &TheModule,
                 StringMap<GlobalValue::LinkageTypes> *ExternalSymbols,
                 SmallPtrSetImpl<const GlobalValue *> &AsmUsed)
      : AsmUndefinedRefs(AsmUndefinedRefs), TM(TM),
        ExternalSymbols(ExternalSymbols), AsmUsed(AsmUsed) {
    accumulateAndSortLibcalls(TheModule);
    for (const Function &F : TheModule)
      findAsmUses(F);
    for (const GlobalVariable &GV : TheModule.globals())
      findAsmUses(GV);
    for (const GlobalAlias &GA : TheModule.aliases())
      findAsmUses(GA);
  }

private:
  // Inputs
  const StringSet<> &AsmUndefinedRefs;
  const TargetMachine &TM;

  // Temps
  llvm::Mangler Mangler;
  std::vector<StringRef> Libcalls;

  // Output
  StringMap<GlobalValue::LinkageTypes> *ExternalSymbols;
  SmallPtrSetImpl<const GlobalValue *> &AsmUsed;

  // Collect names of runtime library functions. User-defined functions with the
  // same names are added to llvm.compiler.used to prevent them from being
  // deleted by optimizations.
  void accumulateAndSortLibcalls(const Module &TheModule) {
    TargetLibraryInfoImpl TLII(Triple(TM.getTargetTriple()));
    TargetLibraryInfo TLI(TLII);

    // TargetLibraryInfo has info on C runtime library calls on the current
    // target.
    for (unsigned I = 0, E = static_cast<unsigned>(LibFunc::NumLibFuncs);
         I != E; ++I) {
      LibFunc::Func F = static_cast<LibFunc::Func>(I);
      if (TLI.has(F))
        Libcalls.push_back(TLI.getName(F));
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
            Libcalls.push_back(Name);
    }

    array_pod_sort(Libcalls.begin(), Libcalls.end());
    Libcalls.erase(std::unique(Libcalls.begin(), Libcalls.end()),
                   Libcalls.end());
  }

  void findAsmUses(const GlobalValue &GV) {
    // There are no restrictions to apply to declarations.
    if (GV.isDeclaration())
      return;

    // There is nothing more restrictive than private linkage.
    if (GV.hasPrivateLinkage())
      return;

    SmallString<64> Buffer;
    TM.getNameWithPrefix(Buffer, &GV, Mangler);

    if (AsmUndefinedRefs.count(Buffer))
      AsmUsed.insert(&GV);

    // Conservatively append user-supplied runtime library functions to
    // llvm.compiler.used.  These could be internalized and deleted by
    // optimizations like -globalopt, causing problems when later optimizations
    // add new library calls (e.g., llvm.memset => memset and printf => puts).
    // Leave it to the linker to remove any dead code (e.g. with -dead_strip).
    if (isa<Function>(GV) &&
        std::binary_search(Libcalls.begin(), Libcalls.end(), GV.getName()))
      AsmUsed.insert(&GV);

    // Record the linkage type of non-local symbols so they can be restored
    // prior
    // to module splitting.
    if (ExternalSymbols && !GV.hasAvailableExternallyLinkage() &&
        !GV.hasLocalLinkage() && GV.hasName())
      ExternalSymbols->insert(std::make_pair(GV.getName(), GV.getLinkage()));
  }
};

} // namespace anonymous

static void findUsedValues(GlobalVariable *LLVMUsed,
                           SmallPtrSetImpl<const GlobalValue *> &UsedValues) {
  if (!LLVMUsed)
    return;

  ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
  for (unsigned i = 0, e = Inits->getNumOperands(); i != e; ++i)
    if (GlobalValue *GV =
            dyn_cast<GlobalValue>(Inits->getOperand(i)->stripPointerCasts()))
      UsedValues.insert(GV);
}

// mark which symbols can not be internalized
void llvm::LTOInternalize(
    Module &TheModule, const TargetMachine &TM,
    const std::function<bool(const GlobalValue &)> &MustPreserveSymbols,
    const StringSet<> &AsmUndefinedRefs,
    StringMap<GlobalValue::LinkageTypes> *ExternalSymbols) {
  SmallPtrSet<const GlobalValue *, 8> AsmUsed;
  ComputeAsmUsed(AsmUndefinedRefs, TM, TheModule, ExternalSymbols, AsmUsed);

  GlobalVariable *LLVMCompilerUsed =
      TheModule.getGlobalVariable("llvm.compiler.used");
  findUsedValues(LLVMCompilerUsed, AsmUsed);
  if (LLVMCompilerUsed)
    LLVMCompilerUsed->eraseFromParent();

  if (!AsmUsed.empty()) {
    llvm::Type *i8PTy = llvm::Type::getInt8PtrTy(TheModule.getContext());
    std::vector<Constant *> asmUsed2;
    for (const auto *GV : AsmUsed) {
      Constant *c =
          ConstantExpr::getBitCast(const_cast<GlobalValue *>(GV), i8PTy);
      asmUsed2.push_back(c);
    }

    llvm::ArrayType *ATy = llvm::ArrayType::get(i8PTy, asmUsed2.size());
    LLVMCompilerUsed = new llvm::GlobalVariable(
        TheModule, ATy, false, llvm::GlobalValue::AppendingLinkage,
        llvm::ConstantArray::get(ATy, asmUsed2), "llvm.compiler.used");

    LLVMCompilerUsed->setSection("llvm.metadata");
  }

  internalizeModule(TheModule, MustPreserveSymbols);
}
