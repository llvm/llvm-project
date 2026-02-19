set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Reducer)
add_mlir_generic_tablegen_target(MLIRReducerIncGen)

set(LLVM_TARGET_DEFINITIONS DialectReductionPatternInterface.td)
mlir_tablegen(DialectReductionPatternInterface.h.inc -gen-dialect-interface-decls)
add_mlir_generic_tablegen_target(MLIRDialectReductionPatternInterfaceIncGen)

add_mlir_doc(Passes ReducerPasses ./ -gen-pass-doc)
