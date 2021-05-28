//===- ForceFunctionAttrs.cpp - Force function attrs for debugging --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO/ForceFunctionAttrs.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define DEBUG_TYPE "forceattrs"

static cl::list<std::string>
    ForceAttributes("force-attribute", cl::Hidden,
                    cl::desc("Add an attribute to a function. This should be a "
                             "pair of 'function-name:attribute-name', for "
                             "example -force-attribute=foo:noinline. This "
                             "option can be specified multiple times."));

static cl::list<std::string> ForceRemoveAttributes(
    "force-remove-attribute", cl::Hidden,
    cl::desc("Remove an attribute from a function. This should be a "
             "pair of 'function-name:attribute-name', for "
             "example -force-remove-attribute=foo:noinline. This "
             "option can be specified multiple times."));

static Attribute::AttrKind parseAttrKind(StringRef Kind) {
  return StringSwitch<Attribute::AttrKind>(Kind)
      .Case("alwaysinline", Attribute::AlwaysInline)
      .Case("builtin", Attribute::Builtin)
      .Case("cold", Attribute::Cold)
      .Case("convergent", Attribute::Convergent)
      .Case("inlinehint", Attribute::InlineHint)
      .Case("jumptable", Attribute::JumpTable)
      .Case("minsize", Attribute::MinSize)
      .Case("naked", Attribute::Naked)
      .Case("nobuiltin", Attribute::NoBuiltin)
      .Case("noduplicate", Attribute::NoDuplicate)
      .Case("noimplicitfloat", Attribute::NoImplicitFloat)
      .Case("noinline", Attribute::NoInline)
      .Case("nonlazybind", Attribute::NonLazyBind)
      .Case("noredzone", Attribute::NoRedZone)
      .Case("noreturn", Attribute::NoReturn)
      .Case("nocf_check", Attribute::NoCfCheck)
      .Case("norecurse", Attribute::NoRecurse)
      .Case("nounwind", Attribute::NoUnwind)
      .Case("nosanitize_coverage", Attribute::NoSanitizeCoverage)
      .Case("optforfuzzing", Attribute::OptForFuzzing)
      .Case("optnone", Attribute::OptimizeNone)
      .Case("optsize", Attribute::OptimizeForSize)
      .Case("readnone", Attribute::ReadNone)
      .Case("readonly", Attribute::ReadOnly)
      .Case("argmemonly", Attribute::ArgMemOnly)
      .Case("returns_twice", Attribute::ReturnsTwice)
      .Case("safestack", Attribute::SafeStack)
      .Case("shadowcallstack", Attribute::ShadowCallStack)
      .Case("sanitize_address", Attribute::SanitizeAddress)
      .Case("sanitize_hwaddress", Attribute::SanitizeHWAddress)
      .Case("sanitize_memory", Attribute::SanitizeMemory)
      .Case("sanitize_thread", Attribute::SanitizeThread)
      .Case("sanitize_memtag", Attribute::SanitizeMemTag)
      .Case("speculative_load_hardening", Attribute::SpeculativeLoadHardening)
      .Case("ssp", Attribute::StackProtect)
      .Case("sspreq", Attribute::StackProtectReq)
      .Case("sspstrong", Attribute::StackProtectStrong)
      .Case("strictfp", Attribute::StrictFP)
      .Case("uwtable", Attribute::UWTable)
      .Case("vscale_range", Attribute::VScaleRange)
      .Default(Attribute::None);
}

/// If F has any forced attributes given on the command line, add them.
/// If F has any forced remove attributes given on the command line, remove
/// them. When both force and force-remove are given to a function, the latter
/// takes precedence.
static void forceAttributes(Function &F) {
  auto ParseFunctionAndAttr = [&](StringRef S) {
    auto Kind = Attribute::None;
    auto KV = StringRef(S).split(':');
    if (KV.first != F.getName())
      return Kind;
    Kind = parseAttrKind(KV.second);
    if (Kind == Attribute::None) {
      LLVM_DEBUG(dbgs() << "ForcedAttribute: " << KV.second
                        << " unknown or not handled!\n");
    }
    return Kind;
  };

  for (auto &S : ForceAttributes) {
    auto Kind = ParseFunctionAndAttr(S);
    if (Kind == Attribute::None || F.hasFnAttribute(Kind))
      continue;
    F.addFnAttr(Kind);
  }

  for (auto &S : ForceRemoveAttributes) {
    auto Kind = ParseFunctionAndAttr(S);
    if (Kind == Attribute::None || !F.hasFnAttribute(Kind))
      continue;
    F.removeFnAttr(Kind);
  }
}

static bool hasForceAttributes() {
  return !ForceAttributes.empty() || !ForceRemoveAttributes.empty();
}

PreservedAnalyses ForceFunctionAttrsPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  if (!hasForceAttributes())
    return PreservedAnalyses::all();

  for (Function &F : M.functions())
    forceAttributes(F);

  // Just conservatively invalidate analyses, this isn't likely to be important.
  return PreservedAnalyses::none();
}

namespace {
struct ForceFunctionAttrsLegacyPass : public ModulePass {
  static char ID; // Pass identification, replacement for typeid
  ForceFunctionAttrsLegacyPass() : ModulePass(ID) {
    initializeForceFunctionAttrsLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override {
    if (!hasForceAttributes())
      return false;

    for (Function &F : M.functions())
      forceAttributes(F);

    // Conservatively assume we changed something.
    return true;
  }
};
}

char ForceFunctionAttrsLegacyPass::ID = 0;
INITIALIZE_PASS(ForceFunctionAttrsLegacyPass, "forceattrs",
                "Force set function attributes", false, false)

Pass *llvm::createForceFunctionAttrsLegacyPass() {
  return new ForceFunctionAttrsLegacyPass();
}
