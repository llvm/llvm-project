set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SCF)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix SCF)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix SCF)
add_public_tablegen_target(MLIRSCFPassIncGen)
add_dependencies(mlir-headers MLIRSCFPassIncGen)

add_mlir_doc(Passes SCFPasses ./ -gen-pass-doc)

