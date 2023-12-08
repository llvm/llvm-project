set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name MLProgram)
add_public_tablegen_target(MLIRMLProgramPassIncGen)
add_dependencies(mlir-headers MLIRMLProgramPassIncGen)

add_mlir_doc(Passes MLProgramPasses ./ -gen-pass-doc)
