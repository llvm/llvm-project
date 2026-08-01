//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "WebAssembly.h"
#include "WebAssemblyTargetMachine.h"
#include "llvm/IR/Analysis.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Scalar/LowerAtomicPass.h"

using namespace llvm;

#define DEBUG_TYPE "wasm-coalesce-features-and-strip-atomics"

namespace {
class WebAssemblyCoalesceFeaturesAndStripAtomicsLegacy final
    : public ModulePass {
  // Take the union of all features used in the module and use it for each
  // function individually, since having multiple feature sets in one module
  // currently does not make sense for WebAssembly. If atomics are not enabled,
  // also strip atomic operations and thread local storage.
  WebAssemblyTargetMachine *WasmTM;

public:
  static char ID;

  WebAssemblyCoalesceFeaturesAndStripAtomicsLegacy(
      WebAssemblyTargetMachine *WasmTM)
      : ModulePass(ID), WasmTM(WasmTM) {}

  bool runOnModule(Module &M) override;
};
} // namespace

char WebAssemblyCoalesceFeaturesAndStripAtomicsLegacy::ID = 0;
INITIALIZE_PASS(WebAssemblyCoalesceFeaturesAndStripAtomicsLegacy, DEBUG_TYPE,
                "Coalesce features and strip atomics", true, false)

ModulePass *llvm::createWebAssemblyCoalesceFeaturesAndStripAtomicsLegacyPass(
    WebAssemblyTargetMachine &TM) {
  return new WebAssemblyCoalesceFeaturesAndStripAtomicsLegacy(&TM);
}

static std::string getFeatureString(const WebAssemblySubtarget *ST,
                                    const FeatureBitset &Features) {
  std::string Ret;
  for (const SubtargetFeatureKV &KV : ST->getAllProcessorFeatures()) {
    if (Features[KV.Value])
      Ret += (StringRef("+") + KV.key() + ",").str();
    else
      Ret += (StringRef("-") + KV.key() + ",").str();
  }
  // remove trailing ','
  Ret.pop_back();
  return Ret;
}

static std::pair<FeatureBitset, std::string>
coalesceFeatures(const Module &M, WebAssemblyTargetMachine *WasmTM) {
  // Union the features of all defined functions. Start with an empty set, so
  // that if a feature is disabled in every function, we'll compute it as
  // disabled. If any function lacks a target-features attribute, it'll
  // default to the target CPU from the `TargetMachine`.
  FeatureBitset Features;
  // We need any MCSubtargetInfo to access WebAssemblyFeatureKV.
  const WebAssemblySubtarget *AnyST = nullptr;
  for (auto &F : M) {
    if (F.isDeclaration())
      continue;

    AnyST = WasmTM->getSubtargetImpl(F);
    Features |= AnyST->getFeatureBits();
  }

  // If we have no defined functions, use the target CPU from the
  // `TargetMachine`.
  if (!AnyST) {
    AnyST =
        WasmTM->getSubtargetImpl(std::string(WasmTM->getTargetCPU()),
                                 std::string(WasmTM->getTargetFeatureString()));
    Features = AnyST->getFeatureBits();
  }

  return {Features, getFeatureString(AnyST, Features)};
}

static void replaceFeatures(Function &F, const std::string &Features) {
  F.removeFnAttr("target-features");
  F.removeFnAttr("target-cpu");
  F.addFnAttr("target-features", Features);
}

static bool stripAtomics(Module &M) {
  // Detect whether any atomics will be lowered, since there is no way to tell
  // whether the LowerAtomic pass lowers e.g. stores.
  bool Stripped = false;
  for (auto &F : M) {
    for (auto &B : F) {
      for (auto &I : B) {
        if (I.isAtomic()) {
          Stripped = true;
          goto done;
        }
      }
    }
  }

done:
  if (!Stripped)
    return false;

  LowerAtomicPass Lowerer;
  FunctionAnalysisManager FAM;
  for (auto &F : M)
    Lowerer.run(F, FAM);

  return true;
}

static bool stripThreadLocals(Module &M) {
  bool Stripped = false;
  for (auto &GV : M.globals()) {
    if (GV.isThreadLocal()) {
      // replace `@llvm.threadlocal.address.pX(GV)` with `GV`.
      for (Use &U : make_early_inc_range(GV.uses())) {
        if (IntrinsicInst *II = dyn_cast<IntrinsicInst>(U.getUser())) {
          if (II->getIntrinsicID() == Intrinsic::threadlocal_address &&
              II->getArgOperand(0) == &GV) {
            II->replaceAllUsesWith(&GV);
            II->eraseFromParent();
          }
        }
      }

      Stripped = true;
      GV.setThreadLocal(false);
    }
  }
  return Stripped;
}

static void recordFeatures(Module &M, const WebAssemblySubtarget *ST,
                           const FeatureBitset &Features, bool Stripped) {
  for (const SubtargetFeatureKV &KV : ST->getAllProcessorFeatures()) {
    if (Features[KV.Value]) {
      // Mark features as used
      std::string MDKey = (StringRef("wasm-feature-") + KV.key()).str();
      M.addModuleFlag(Module::ModFlagBehavior::Error, MDKey,
                      wasm::WASM_FEATURE_PREFIX_USED);
    }
  }
  // Code compiled without atomics or bulk-memory may have had its atomics or
  // thread-local data lowered to nonatomic operations or non-thread-local
  // data. In that case, we mark the pseudo-feature "shared-mem" as disallowed
  // to tell the linker that it would be unsafe to allow this code to be used
  // in a module with shared memory.
  if (Stripped) {
    M.addModuleFlag(Module::ModFlagBehavior::Error, "wasm-feature-shared-mem",
                    wasm::WASM_FEATURE_PREFIX_DISALLOWED);
  }
}

static bool coalesceFeaturesAndStripAtomics(Module &M,
                                            WebAssemblyTargetMachine *WasmTM) {
  auto [Features, FeatureStr] = coalesceFeatures(M, WasmTM);

  WasmTM->setTargetFeatureString(FeatureStr);
  for (auto &F : M)
    replaceFeatures(F, FeatureStr);

  bool StrippedAtomics = false;
  bool StrippedTLS = false;

  // In cooperative threading mode, thread locals are meaningful even without
  // atomics.
  const WebAssemblySubtarget *ST = WasmTM->getSubtargetImpl();
  bool CooperativeThreading = ST->hasCooperativeMultithreading();

  if (!Features[WebAssembly::FeatureAtomics]) {
    StrippedAtomics = stripAtomics(M);
    if (!CooperativeThreading)
      StrippedTLS = stripThreadLocals(M);
  }
  if (!Features[WebAssembly::FeatureBulkMemory] && !StrippedTLS) {
    StrippedTLS = stripThreadLocals(M);
  }

  if (StrippedAtomics && !StrippedTLS && !CooperativeThreading)
    stripThreadLocals(M);
  else if (StrippedTLS && !StrippedAtomics)
    stripAtomics(M);

  recordFeatures(M, ST, Features, StrippedAtomics || StrippedTLS);

  // Conservatively assume we have made some change
  return true;
}

bool WebAssemblyCoalesceFeaturesAndStripAtomicsLegacy::runOnModule(Module &M) {
  return coalesceFeaturesAndStripAtomics(M, WasmTM);
}

PreservedAnalyses WebAssemblyCoalesceFeaturesAndStripAtomicsPass::run(
    Module &M, ModuleAnalysisManager &MAM) {
  return coalesceFeaturesAndStripAtomics(M, &TM) ? PreservedAnalyses::none()
                                                 : PreservedAnalyses::all();
}
