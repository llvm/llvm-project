set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Reducer)
add_mlir_generic_tablegen_target(MLIRReducerIncGen)

add_mlir_doc(Passes ReducerPasses ./ -gen-pass-doc)

set(LLVM_TARGET_DEFINITIONS ReductionPatternInterface.td)
mlir_tablegen(ReductionPatternInterface.h.inc -gen-dialect-interface-decls)
add_mlir_generic_tablegen_target(MLIRReductionPatternInterfaceGen)
