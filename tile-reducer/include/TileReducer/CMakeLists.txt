add_mlir_dialect(TileReducerOps tr)
add_mlir_doc(TileReducerDialect TileReducerDialect TileReducer/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TileReducerPasses.td)
mlir_tablegen(TileReducerPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTileReducerPassesIncGen)
