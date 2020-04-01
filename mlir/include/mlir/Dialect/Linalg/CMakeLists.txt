add_subdirectory(IR)
add_subdirectory(Transforms)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRLinalgPassIncGen)
