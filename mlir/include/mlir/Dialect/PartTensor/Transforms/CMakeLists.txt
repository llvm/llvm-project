set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name PartTensor)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix PartTensor)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix PartTensor)
add_public_tablegen_target(MLIRPartTensorPassIncGen)

add_mlir_doc(Passes PartTensorPasses ./ -gen-pass-doc)
