set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OMP)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix OMP)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix OMP)
add_public_tablegen_target(MLIROMPPassIncGen)

add_mlir_doc(Passes OMPPasses ./ -gen-pass-doc)
