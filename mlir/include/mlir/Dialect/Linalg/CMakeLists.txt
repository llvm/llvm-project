add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Linalg)
add_public_tablegen_target(MLIRLinalgPassIncGen)

add_mlir_doc(Passes -gen-pass-doc LinalgPasses ./)
