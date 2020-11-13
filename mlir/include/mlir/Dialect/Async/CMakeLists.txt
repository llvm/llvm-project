add_subdirectory(IR)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Async)
add_public_tablegen_target(MLIRAsyncPassIncGen)

add_mlir_doc(Passes -gen-pass-doc AsyncPasses ./)
