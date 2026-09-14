//===- InjectOpenMPVFABIMappings.cpp - OpenMP _ZGV to VFABI conversion ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts `_ZGV...` function attributes (emitted by OMPIRBuilder for OpenMP
// `declare simd` functions) into `vector-function-abi-variant` attribute that
// LoopVectorize / VFDatabase consumes.
//
// OMPIRBuilder emits one function attribute per vector variant:
//
//   attributes #0 = { "_ZGVnN4v_foo" "_ZGVnM4v_foo" }
//
// VFDatabase::getMappings() reads the `vector-function-abi-variant`
// attribute (via CallBase::getFnAttr fallthrough to callee attrs):
//
//   attributes #0 = {
//     "vector-function-abi-variant"=
//       "_ZGVnN4v_foo(_ZGVnN4v_foo),_ZGVnM4v_foo(_ZGVnM4v_foo)" }
//
// This pass bridges the two by:
//   1. Collecting all `_ZGV*` attrs from each function.
//   2. Demangling each to a VFInfo via VFABI::tryDemangleForVFABI.
//   3. Creating external declarations for the vector variants (so
//      VFDatabase can verify they exist via Module::getFunction).
//   4. Consolidating the names into a single `vector-function-abi-variant`
//      attribute and removing the `_ZGV...` attrs.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/InjectOpenMPVFABIMappings.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/AttributeMask.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/VFABIDemangler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

#define DEBUG_TYPE "inject-openmp-vfabi-mappings"

using namespace llvm;

STATISTIC(NumFunctionsInjected,
          "Number of functions with VFABI mappings injected.");
STATISTIC(NumVFDeclAdded, "Number of vector function declarations added.");
STATISTIC(NumCompUsedAdded,
          "Number of `@llvm.compiler.used` operands that have been added.");

// Convert `_ZGV` attrs into `vector-function-abi-variant` and emit
// vector declarations. Returns true if the function was modified.
static bool
injectMappingsForFunction(Function &F,
                          SmallVectorImpl<GlobalValue *> &NewDecls) {
  Module *M = F.getParent();
  FunctionType *ScalarFTy = F.getFunctionType();

  // Collect raw _ZGV attrs from the function.
  SmallVector<std::string, 8> MangledNames;
  for (const Attribute &A : F.getAttributes().getFnAttrs()) {
    if (!A.isStringAttribute())
      continue;
    StringRef Key = A.getKindAsString();
    if (!Key.starts_with("_ZGV"))
      continue;
    MangledNames.push_back(Key.str());
  }
  if (MangledNames.empty())
    return false;

  // Preserve any pre-existing vector-function-abi-variant mappings.
  SmallVector<std::string, 8> Mappings;
  StringRef Existing =
      F.getFnAttribute(VFABI::MappingsAttrName).getValueAsString();
  if (!Existing.empty()) {
    SmallVector<StringRef, 8> ExistingPieces;
    Existing.split(ExistingPieces, ',');
    for (const auto &P : ExistingPieces)
      Mappings.push_back(P.str());
  }

  // Build VFABI mapping strings and vector declarations.
  for (const std::string &Name : MangledNames) {
    std::optional<VFInfo> Info = VFABI::tryDemangleForVFABI(Name, ScalarFTy);
    if (!Info)
      continue;

    // Build the VFABI mapping: "_ZGVnN4v_foo(_ZGVnN4v_foo)"
    Mappings.push_back(Name + "(" + Info->VectorName + ")");

    // Add vector function declaration if it doesn't already exist.
    if (!M->getFunction(Info->VectorName)) {
      FunctionType *VecFTy = VFABI::createFunctionType(*Info, ScalarFTy);
      Function *VecFunc = Function::Create(VecFTy, Function::ExternalLinkage,
                                           Info->VectorName, M);

      // Copy attributes from the scalar function and strip ones that are
      // incompatible with the vectorized types (e.g., signext on vectors).
      VecFunc->copyAttributesFrom(&F);
      VecFunc->removeRetAttrs(AttributeFuncs::typeIncompatible(
          VecFunc->getReturnType(), VecFunc->getAttributes().getRetAttrs()));
      for (auto &Arg : VecFunc->args())
        Arg.removeAttrs(AttributeFuncs::typeIncompatible(Arg.getType(),
                                                         Arg.getAttributes()));
      // Remove _ZGV attrs that were copied from the scalar function.
      for (const std::string &N : MangledNames)
        VecFunc->removeFnAttr(N);
      // The vector declaration does not need the mapping attribute itself.
      VecFunc->removeFnAttr(VFABI::MappingsAttrName);

      // Set the appropriate calling convention for the vector variant.
      // The Clang/Flang declare simd codegen only targets AArch64 and x86.
      // AArch64 NEON uses aarch64_vector_pcs and SVE uses
      // aarch64_sve_vector_pcs. x86 ISAs (SSE, AVX, AVX2, AVX512) use the
      // default C calling convention, consistent with VecFuncs.def.
      switch (Info->ISA) {
      case VFISAKind::AdvancedSIMD:
        VecFunc->setCallingConv(CallingConv::AArch64_VectorCall);
        break;
      case VFISAKind::SVE:
        VecFunc->setCallingConv(CallingConv::AArch64_SVE_VectorCall);
        break;
      default:
        break;
      }

      LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added to the module: `"
                        << Info->VectorName << "` of type " << *VecFTy << "\n");
      ++NumVFDeclAdded;

      NewDecls.push_back(VecFunc);
    }

    // Remove "_ZGV...".
    F.removeFnAttr(Name);
  }

  // Set the consolidated vector-function-abi-variant attribute.
  if (!Mappings.empty()) {
    SmallString<256> Buffer;
    raw_svector_ostream Out(Buffer);
    interleave(Mappings, Out, [&](const std::string &S) { Out << S; }, ",");
    F.addFnAttr(VFABI::MappingsAttrName, Buffer.str());
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Set `" << VFABI::MappingsAttrName
                      << "` for `" << F.getName() << "` to `" << Buffer
                      << "`\n");
    ++NumFunctionsInjected;
  }

  return true;
}

PreservedAnalyses InjectOpenMPVFABIMappings::run(Module &M,
                                                 ModuleAnalysisManager &AM) {
  bool Changed = false;
  SmallVector<GlobalValue *, 8> NewDecls;
  for (Function &F : M)
    Changed |= injectMappingsForFunction(F, NewDecls);

  // Batch-add all new declarations to @llvm.compiler.used.
  if (!NewDecls.empty()) {
    appendToCompilerUsed(M, NewDecls);
    NumCompUsedAdded += NewDecls.size();
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added " << NewDecls.size()
                      << " declarations to `@llvm.compiler.used`.\n");
  }

  if (!Changed)
    return PreservedAnalyses::all();

  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
