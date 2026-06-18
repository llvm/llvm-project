add_mlir_dialect(SCFOps scf Ops)
add_mlir_doc(SCFOps SCFDialect Dialects/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS DeviceMappingInterface.td)
mlir_tablegen(DeviceMappingAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(DeviceMappingAttrInterface.cpp.inc -gen-attr-interface-defs)
mlir_tablegen(DeviceMappingAttributes.h.inc -gen-attrdef-decls)
mlir_tablegen(DeviceMappingAttributes.cpp.inc -gen-attrdef-defs)
add_mlir_generic_tablegen_target(MLIRDeviceMappingInterfacesIncGen)
