set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Tensor)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Tensor)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Tensor)
add_mlir_dialect_tablegen_target(MLIRTensorTransformsIncGen)

add_mlir_doc(Passes TensorPasses ./ -gen-pass-doc)
