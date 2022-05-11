#ifndef QUANTUM_PASSES_TD
#define QUANTUM_PASSES_TD

include "mlir/Pass/PassBase.td"

// Transforms/optimizations
def QuantumRewritePass : FunctionPass<"qssa-apply-rewrites"> {
  let summary = "Apply basic qssa rewrite rules";
  let constructor = "mlir::createQuantumRewritePass()";
}

def QuantumConvert1QToUPass : FunctionPass<"qssa-convert-1q-to-U"> {
  let summary = "Convert all single qubit primitives to U (u3) calls";
  let constructor = "mlir::createQuantumConvert1QToUPass()";
}

def QuantumPrepareForZXPass : FunctionPass<"qssa-prepare-for-zx"> {
  let summary = "Replace all gates with ZX compatible gates - Rx, Rz, CNOT";
  let constructor = "mlir::createQuantumPrepareForZXPass()";
}

// Dynamic Analysis (with updates)
def QuantumDepthComputePass : FunctionPass<"qssa-compute-depths"> {
  let summary = "Compute circuit depth for all operations";
  let constructor = "mlir::createQuantumDepthComputePass()";
}

def QuantumDepthClearPass : FunctionPass<"qssa-clear-depths"> {
  let summary = "Clear all computed depth attributes";
  let constructor = "mlir::createQuantumClearDepthPass()";
}

// Static Analysis (no updates)
def QuantumGateCountPass : Pass<"qssa-gate-count", "ModuleOp"> {
  let summary = "Count gates and sub-circuit patterns in main (marked qasm.main)";
  let constructor = "mlir::createQuantumGateCountPass()";
}

#endif // QUANTUM_PASSES_TD
