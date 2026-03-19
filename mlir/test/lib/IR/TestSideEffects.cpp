//===- TestSidEffects.cpp - Pass to test side effects ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace {
struct SideEffectsPass
    : public PassWrapper<SideEffectsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SideEffectsPass)

  StringRef getArgument() const final { return "test-side-effects"; }
  StringRef getDescription() const final {
    return "Test side effects interfaces";
  }
  void runOnOperation() override {
    auto module = getOperation();

    // Walk operations detecting side effects.
    SmallVector<MemoryEffects::EffectInstance, 8> effects;
    module.walk([&](MemoryEffectOpInterface op) {
      effects.clear();
      op.getEffects(effects);

      if (op->hasTrait<OpTrait::IsTerminator>()) {
        return;
      }

      // Check to see if this operation has any memory effects.
      if (effects.empty()) {
        op.emitRemark() << "operation has no memory effects";
        return;
      }

      for (MemoryEffects::EffectInstance instance : effects) {
        auto diag = op.emitRemark() << "found an instance of ";

        if (isa<MemoryEffects::Allocate>(instance.getEffect()))
          diag << "'allocate'";
        else if (isa<MemoryEffects::Free>(instance.getEffect()))
          diag << "'free'";
        else if (isa<MemoryEffects::Read>(instance.getEffect()))
          diag << "'read'";
        else if (isa<MemoryEffects::Write>(instance.getEffect()))
          diag << "'write'";

        if (instance.getValue()) {
          if (auto *opOpd = instance.getEffectValue<OpOperand *>())
            diag << " on op operand " << opOpd->getOperandNumber() << ",";
          else if (auto opRes = instance.getEffectValue<OpResult>())
            diag << " on op result " << opRes.getResultNumber() << ",";
          else if (auto opBlk = instance.getEffectValue<BlockArgument>())
            diag << " on block argument " << opBlk.getArgNumber() << ",";
        } else if (SymbolRefAttr symbolRef = instance.getSymbolRef())
          diag << " on a symbol '" << symbolRef << "',";

        diag << " on resource '" << instance.getResource()->getName() << "'";
      }
    });

    SmallVector<TestEffects::EffectInstance, 1> testEffects;
    module.walk([&](TestEffectOpInterface op) {
      testEffects.clear();
      op.getEffects(testEffects);

      if (testEffects.empty())
        return;

      for (const TestEffects::EffectInstance &instance : testEffects) {
        op.emitRemark() << "found a parametric effect with "
                        << instance.getParameters();
      }
    });
  }
};
/// External model that attaches MemoryEffectOpInterface to
/// ExternalSideEffectOp. The interface is optional: hasKnownMemoryEffects
/// returns true only when the "effects" attribute is present.
struct ExternalSideEffectModel
    : public MemoryEffectOpInterface::ExternalModel<
          ExternalSideEffectModel, test::ExternalSideEffectOp> {
  void
  getEffects(Operation *op,
             SmallVectorImpl<MemoryEffects::EffectInstance> &effects) const {
    ArrayAttr effectsAttr = op->getAttrOfType<ArrayAttr>("effects");
    if (!effectsAttr)
      return;
    for (Attribute element : effectsAttr) {
      DictionaryAttr effectElement = cast<DictionaryAttr>(element);
      MemoryEffects::Effect *effect =
          StringSwitch<MemoryEffects::Effect *>(
              cast<StringAttr>(effectElement.get("effect")).getValue())
              .Case("allocate", MemoryEffects::Allocate::get())
              .Case("free", MemoryEffects::Free::get())
              .Case("read", MemoryEffects::Read::get())
              .Case("write", MemoryEffects::Write::get());
      effects.emplace_back(effect, SideEffects::DefaultResource::get());
    }
  }

  bool hasKnownMemoryEffects(Operation *op) const {
    return op->hasAttr("effects");
  }
};

struct ExternalSideEffectsPass
    : public PassWrapper<ExternalSideEffectsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ExternalSideEffectsPass)

  StringRef getArgument() const final { return "test-external-side-effects"; }
  StringRef getDescription() const final {
    return "Test external model for MemoryEffectsOpInterface with "
           "hasKnownMemoryEffects";
  }
  void runOnOperation() override {
    auto module = getOperation();

    // Attach the external model.
    test::ExternalSideEffectOp::attachInterface<ExternalSideEffectModel>(
        *module.getContext());

    module.walk([&](Operation *op) {
      if (op->hasTrait<OpTrait::IsTerminator>())
        return;
      if (!isa<test::ExternalSideEffectOp>(op))
        return;

      bool implements = isa<MemoryEffectOpInterface>(op);
      op->emitRemark() << "implements MemoryEffectOpInterface: "
                       << (implements ? "true" : "false");

      if (implements) {
        auto iface = cast<MemoryEffectOpInterface>(op);
        SmallVector<MemoryEffects::EffectInstance, 8> effects;
        iface.getEffects(effects);
        if (effects.empty()) {
          op->emitRemark() << "operation has no memory effects";
        }
        for (auto &instance : effects) {
          auto diag = op->emitRemark() << "found an instance of ";
          if (isa<MemoryEffects::Read>(instance.getEffect()))
            diag << "'read'";
          else if (isa<MemoryEffects::Write>(instance.getEffect()))
            diag << "'write'";
          diag << " on resource '" << instance.getResource()->getName() << "'";
        }
      }
    });
  }
};

} // namespace

namespace mlir {
void registerSideEffectTestPasses() {
  PassRegistration<SideEffectsPass>();
  PassRegistration<ExternalSideEffectsPass>();
}
} // namespace mlir
