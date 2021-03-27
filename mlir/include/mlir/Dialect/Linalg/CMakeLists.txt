add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Linalg)
mlir_tablegen(Passes.capi.h.inc -gen-pass-capi-header --prefix Linalg)
mlir_tablegen(Passes.capi.cpp.inc -gen-pass-capi-impl --prefix Linalg)
add_public_tablegen_target(MLIRLinalgPassIncGen)

add_mlir_doc(Passes -gen-pass-doc LinalgPasses ./)
