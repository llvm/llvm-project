#ifndef QUANTUM_PASSES_H_
#define QUANTUM_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

class Pass;

std::unique_ptr<FunctionPass> createQuantumRewritePass();
std::unique_ptr<FunctionPass> createQuantumConvert1QToUPass();
std::unique_ptr<FunctionPass> createQuantumPrepareForZXPass();

std::unique_ptr<FunctionPass> createQuantumDepthComputePass();
std::unique_ptr<FunctionPass> createQuantumClearDepthPass();
std::unique_ptr<Pass> createQuantumGateCountPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Dialect/Quantum/Passes.h.inc"

} // namespace mlir

#endif // QUANTUM_PASSES_H_
