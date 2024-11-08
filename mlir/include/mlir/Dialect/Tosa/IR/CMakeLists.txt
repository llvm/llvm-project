add_mlir_dialect(TosaOps tosa)
add_mlir_doc(TosaOps TosaOps Dialects/ -gen-op-doc)
add_mlir_interface(TosaInterfaces)

set(LLVM_TARGET_DEFINITIONS TosaOps.td)
mlir_tablegen(TosaAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=tosa)
mlir_tablegen(TosaAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=tosa)
add_public_tablegen_target(MLIRTosaAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS TosaDialectBytecode.td)
mlir_tablegen(TosaDialectBytecode.cpp.inc -gen-bytecode -bytecode-dialect="Tosa")
add_public_tablegen_target(MLIRTosaDialectBytecodeIncGen)

