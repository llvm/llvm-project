#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToEmitC/SCFToEmitC.h"
#include "mlir/Conversion/ArithToEmitC/ArithToEmitC.h"
#include "mlir/Conversion/MathToEmitC/MathToEmitC.h"
#include "mlir/Conversion/MemRefToEmitC/MemRefToEmitC.h"
#include "mlir/Conversion/ConvertToEmitC/ConvertToEmitCPass.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

struct TTLToEmitC : public PassWrapper<TTLToEmitC, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TTLToEmitC)

  TTLToEmitC() = default;
  TTLToEmitC(const TTLToEmitC &other) : PassWrapper<TTLToEmitC, OperationPass<ModuleOp>>(other) {}

  StringRef getArgument() const override { return "ttl-to-emitc"; }
  StringRef getDescription() const override { return "Convert TTL operations to EmitC dialect"; }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    PassManager pm(module->getContext());
    
    // First convert Affine to SCF, MemRef, etc...
    pm.addNestedPass<func::FuncOp>(createLowerAffinePass());
    
    // Then convert all dialects to EmitC
    pm.addNestedPass<func::FuncOp>(createConvertArithToEmitC());
    pm.addNestedPass<func::FuncOp>(createConvertMathToEmitC());
    pm.addNestedPass<func::FuncOp>(createConvertMemRefToEmitC());
    pm.addNestedPass<func::FuncOp>(createSCFToEmitC());
    pm.addNestedPass<func::FuncOp>(createConvertToEmitC());
    
    // Clean up passes
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    
    // Reconcile unrealized casts must run at module level
    pm.addPass(createReconcileUnrealizedCastsPass());
    
    if (failed(pm.run(module))) {
      signalPassFailure();
    }
  }
};

void registerTTLToEmitC() {
  PassRegistration<TTLToEmitC>();
}

} // end anonymous namespace

namespace mlir {
std::unique_ptr<Pass> createTTLToEmitC() {
  return std::make_unique<TTLToEmitC>();
}
} // end namespace mlir 