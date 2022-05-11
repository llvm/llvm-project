add_mlir_dialect(QuantumOps qssa)
add_mlir_doc(QuantumDialect QuantumDialect Quantum/ -gen-dialect-doc)
add_mlir_doc(QuantumTypes QuantumTypes Quantum/ -gen-op-doc)
add_mlir_doc(QuantumOps QuantumOps Quantum/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Quantum)
add_public_tablegen_target(MLIRQuantumPassIncGen)

add_subdirectory(Transforms)