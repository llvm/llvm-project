add_mlir_dialect(GPUOps gpu)
add_mlir_doc(GPUOps GPUOps Dialects/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS GPUBase.td)
mlir_tablegen(GPUOpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(GPUOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRGPUOpInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS ParallelLoopMapperAttr.td)
mlir_tablegen(ParallelLoopMapperEnums.h.inc -gen-enum-decls)
mlir_tablegen(ParallelLoopMapperEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRParallelLoopMapperEnumsGen)

set(LLVM_TARGET_DEFINITIONS GPUOps.td)
mlir_tablegen(GPUOpsEnums.h.inc -gen-enum-decls)
mlir_tablegen(GPUOpsEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRGPUOpsEnumsGen)

set(LLVM_TARGET_DEFINITIONS CompilationAttrInterfaces.td)
mlir_tablegen(CompilationAttrInterfaces.h.inc -gen-attr-interface-decls)
mlir_tablegen(CompilationAttrInterfaces.cpp.inc -gen-attr-interface-defs)
add_public_tablegen_target(MLIRGPUCompilationAttrInterfacesIncGen)

set(LLVM_TARGET_DEFINITIONS GPUOps.td)
mlir_tablegen(GPUOpsAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=gpu)
mlir_tablegen(GPUOpsAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=gpu)
add_public_tablegen_target(MLIRGPUOpsAttributesIncGen)
