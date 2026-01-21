set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name OpenMP)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix OpenMP)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix OpenMP)
add_mlir_dialect_tablegen_target(MLIROpenMPPassIncGen)

add_mlir_doc(Passes OpenMPPasses ./ -gen-pass-doc)
