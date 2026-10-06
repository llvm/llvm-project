add_mlir_dialect(TileReducerOps tr)
add_mlir_doc(TileReducerDialect TileReducerDialect TileReducer/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS TileReducerPasses.td)
mlir_tablegen(TileReducerPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRTileReducerPassesIncGen)

set(LLVM_TARGET_DEFINITIONS TileReducerTransformOps.td)
mlir_tablegen(TileReducerTransformOps.h.inc -gen-op-decls)
mlir_tablegen(TileReducerTransformOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRTileReducerTransformOpsIncGen)
