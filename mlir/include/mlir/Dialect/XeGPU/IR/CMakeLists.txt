add_mlir_dialect(XeGPU xegpu)
add_mlir_doc(XeGPU XeGPU Dialects/ -gen-dialect-doc -dialect=xegpu)

set(LLVM_TARGET_DEFINITIONS XeGPU.td)
mlir_tablegen(XeGPUAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=xegpu)
mlir_tablegen(XeGPUAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=xegpu)
add_mlir_dialect_tablegen_target(MLIRXeGPUAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS XeGPUAttrs.td)
mlir_tablegen(XeGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(XeGPUEnums.cpp.inc -gen-enum-defs)
add_mlir_dialect_tablegen_target(MLIRXeGPUEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS XeGPUAttrs.td)
mlir_tablegen(XeGPUAttrInterface.h.inc -gen-attr-interface-decls)
mlir_tablegen(XeGPUAttrInterface.cpp.inc -gen-attr-interface-defs)
add_mlir_dialect_tablegen_target(MLIRXeGPUAttrInterfaceIncGen)
