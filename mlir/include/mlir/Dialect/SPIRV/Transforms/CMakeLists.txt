set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SPIRV)
add_public_tablegen_target(MLIRSPIRVPassIncGen)
add_dependencies(mlir-headers MLIRSPIRVPassIncGen)

add_mlir_doc(Passes SPIRVPasses ./ -gen-pass-doc)

