add_mlir_dialect(XeGPU xegpu)
add_mlir_doc(XeGPU XeGPU Dialects/ -gen-dialect-doc -dialect=xegpu)

set(LLVM_TARGET_DEFINITIONS XeGPU.td)
mlir_tablegen(XeGPUAttrs.h.inc -gen-attrdef-decls -attrdefs-dialect=xegpu)
mlir_tablegen(XeGPUAttrs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=xegpu)
add_public_tablegen_target(MLIRXeGPUAttrsIncGen)
add_dependencies(mlir-headers MLIRXeGPUAttrsIncGen)

set(LLVM_TARGET_DEFINITIONS XeGPUAttrs.td)
mlir_tablegen(XeGPUEnums.h.inc -gen-enum-decls)
mlir_tablegen(XeGPUEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRXeGPUEnumsIncGen)
add_dependencies(mlir-headers MLIRXeGPUEnumsIncGen)
