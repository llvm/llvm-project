add_mlir_interface(OpAsmInterface)
add_mlir_interface(SymbolInterfaces)
add_mlir_interface(RegionKindInterface)

set(LLVM_TARGET_DEFINITIONS BuiltinAttributes.td)
mlir_tablegen(BuiltinAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(BuiltinAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRBuiltinAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS BuiltinDialect.td)
mlir_tablegen(BuiltinDialect.h.inc -gen-dialect-decls)
add_public_tablegen_target(MLIRBuiltinDialectIncGen)

set(LLVM_TARGET_DEFINITIONS BuiltinLocationAttributes.td)
mlir_tablegen(BuiltinLocationAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(BuiltinLocationAttributes.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRBuiltinLocationAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS BuiltinOps.td)
mlir_tablegen(BuiltinOps.h.inc -gen-op-decls)
mlir_tablegen(BuiltinOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRBuiltinOpsIncGen)

set(LLVM_TARGET_DEFINITIONS BuiltinTypes.td)
mlir_tablegen(BuiltinTypes.h.inc -gen-typedef-decls)
mlir_tablegen(BuiltinTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRBuiltinTypesIncGen)

set(LLVM_TARGET_DEFINITIONS TensorEncoding.td)
mlir_tablegen(TensorEncInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(TensorEncInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRTensorEncodingIncGen)

add_mlir_doc(BuiltinAttributes BuiltinAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(BuiltinLocationAttributes BuiltinLocationAttributes Dialects/ -gen-attrdef-doc)
add_mlir_doc(BuiltinOps BuiltinOps Dialects/ -gen-op-doc)
add_mlir_doc(BuiltinTypes BuiltinTypes Dialects/ -gen-typedef-doc)
