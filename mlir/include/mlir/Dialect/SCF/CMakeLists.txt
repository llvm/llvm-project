add_mlir_dialect(SCFOps scf Ops)
add_mlir_doc(SCFOps SCFDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name SCF)
add_public_tablegen_target(MLIRSCFPassIncGen)
add_dependencies(mlir-headers MLIRSCFPassIncGen)

add_mlir_doc(Passes SCFPasses ./ -gen-pass-doc)

add_subdirectory(TransformOps)
