//===- DynamicDebugging.cpp - Dynamic Debugging utils --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/DynamicDebugging.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

using namespace llvm;

std::unique_ptr<Module>
llvm::prepareForDynamicDebugging(Module *M, StringRef PromotionSuffix) {
  using namespace llvm;
  assert(M->getNamedMetadata("llvm.dbg.cu") &&
         "Expected module with debug info");

  auto ShouldPromoteGlobal = [](const GlobalValue &GV) {
    if (!GV.hasLocalLinkage())
      return false;

    // Local symbols in a comdat shouldn't be promoted either.
    // This can happen with (at least) __cxx_global_var_init (which is local
    // and may initialize an ODR-weak global variable).
    if (GV.hasComdat())
      return false;

    return true;
  };

  // Clone functions definitions only - CloneModule will clone data definitions
  // as declarations. We rename these and explicitly set their linkage later.
  auto ShouldCloneDefinition = [](const GlobalValue *GV) {
    return isa<Function>(GV);
  };
  ValueToValueMapTy VMap;
  std::unique_ptr<Module> UnoptM = CloneModule(*M, VMap, ShouldCloneDefinition);

  // Insert declarations into Inner that point to Outer, apply attributes to
  // Outer functions.
  DenseMap<Function *, Function *> OuterDefToInnerDecl;
  for (Function &OuterDef : M->functions()) {
    if (OuterDef.isDeclaration())
      continue;

    // Find the Inner version of Outer's function.
    Function *InnerDef = cast<Function>(VMap[&OuterDef]);

    // Apply some attributes to both Inner and Outer defs.
    {
      // Unoptimized module wants no inlining at all.
      InnerDef->addFnAttr(Attribute::NoInline);
      InnerDef->removeFnAttr(Attribute::AlwaysInline);

      // Apply optnone, remove clashing attributes.
      InnerDef->addFnAttr(Attribute::OptimizeNone);
      InnerDef->removeFnAttr(Attribute::OptimizeForSize);
      InnerDef->removeFnAttr(Attribute::MinSize);

      // Add attributes to the outer-object functions to ensure they're
      // always patchable. TODO: Add patch bytes size/value for other targets.
      if (M->getTargetTriple().isX86_64()) {
        OuterDef.addFnAttr("tail-pad-to-size", "5");
        OuterDef.addFnAttr("tail-pad-value", "144"); // 0x90
        OuterDef.addFnAttr("no-func-spec");
      }
    }

    // Apply COMDAT grouping to the clone if OuterDef is in one.
    if (OuterDef.hasComdat()) {
      std::string NewComdat =
          Twine("__dyndbg." + OuterDef.getComdat()->getName()).str();
      Comdat *C = M->getOrInsertComdat(NewComdat);
      C->setSelectionKind(OuterDef.getComdat()->getSelectionKind());
      InnerDef->setComdat(C);
    }

    // Rename Inner's copy and set appropriate linkage depending on whether
    // it'll get promoted in Outer or not.
    if (ShouldPromoteGlobal(OuterDef)) {
      InnerDef->setName("__dyndbg." + InnerDef->getName() + PromotionSuffix);
      InnerDef->setLinkage(GlobalValue::ExternalLinkage);
      InnerDef->setVisibility(GlobalValue::HiddenVisibility);
    } else {
      InnerDef->setName("__dyndbg." + InnerDef->getName());
      InnerDef->setLinkage(OuterDef.getLinkage());
      InnerDef->setVisibility(OuterDef.getVisibility());
    }

    // Create Inner's external reference to Outer's version.
    Function *InnerDeclOfOuterDef = Function::Create(
        cast<FunctionType>(OuterDef.getValueType()), OuterDef.getLinkage(),
        OuterDef.getAddressSpace(), OuterDef.getName(), UnoptM.get());
    InnerDeclOfOuterDef->copyAttributesFrom(&OuterDef);
    // Re-set linkage and visibility after copyAttributesFrom.
    InnerDeclOfOuterDef->setLinkage(GlobalValue::ExternalLinkage);
    InnerDeclOfOuterDef->setPersonalityFn(nullptr);

    // Replace Inner uses of function with that external reference.
    InnerDef->replaceAllUsesWith(InnerDeclOfOuterDef);

    VMap[&OuterDef] = InnerDeclOfOuterDef;
  }

  // Add Outer aliases for globals with internal linkage, adding
  // ".dyndbg.<TU-unique-hash>" suffix. Update Inner's external references to
  // these promoted functions to use their new names.
  SmallVector<GlobalValue *> GlobalsToPreserve;
  for (GlobalValue &GV : M->global_values()) {
    // If the global is used but may be discarded after optimizations
    // (e.g. inlining) then ensure it's marked as compiler-used to prevent
    // that. It may be referenced from the inner module.
    if (GV.isDiscardableIfUnused()) {
      if (GV.getNumUses()) {
        GlobalsToPreserve.push_back(&GV);
      } else {
        // No uses, so the inner module doesn't need a reference nor do we need
        // to produce an alias.
        // Remove the inner module reference.
        auto GVAndUnoptPair = VMap.find(&GV);
        assert(GVAndUnoptPair != VMap.end() && "Unmapped global?");
        // Delete the external reference - VMap shold only contain mappings to
        // those declarations now.
        assert(cast<GlobalValue>(GVAndUnoptPair->second)->isDeclaration() &&
               "expected only declarations in VMap now");
        cast<GlobalValue>(GVAndUnoptPair->second)->eraseFromParent();
        GVAndUnoptPair->second = nullptr;
        // Nothing else to do for this global.
        continue;
      }
    }

    if (!ShouldPromoteGlobal(GV))
      continue;

    // We need external aliases with a mangled name and hidden visability.
    auto *Alias = GlobalAlias::create(GlobalValue::ExternalLinkage,
                                      GV.getName() + PromotionSuffix, &GV);
    Alias->setVisibility(GlobalValue::HiddenVisibility);

    // Update the Inner external reference that corresponds to the promoted
    // Outer global (created just now as an alias in opt) to reference the new
    // alias.
    GlobalValue *UnoptGV = cast<GlobalValue>(VMap[&GV]);
    UnoptGV->setName(Alias->getName());
    UnoptGV->setVisibility(GlobalValue::HiddenVisibility);
    assert(UnoptGV->getLinkage() == GlobalValue::ExternalLinkage &&
           "Expected ExternalLinkage from CloneModule or inserted decl");
  }

  // Preserve functions that may be discarded after optimizing away call sites
  // (e.g. ODR-weak). Another desirable effect of this is that it prevents
  // GlobalOpt promoting the alias. If the function-preservation mechanism
  // changes in the future GlobalOpt alias promotion must be handled another
  // way.
  appendToCompilerUsed(*M, GlobalsToPreserve);

  return UnoptM;
}