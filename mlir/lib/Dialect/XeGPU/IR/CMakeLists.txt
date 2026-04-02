add_mlir_dialect_library(MLIRXeGPUDialect
  XeGPUDialect.cpp
  XeGPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/mlir/Dialect/XeGPU

  DEPENDS
  MLIRXeGPUIncGen
  MLIRXeGPUAttrInterfaceIncGen
  MLIRXeGPUAttrsIncGen
  MLIRXeGPUEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIndexDialect
  MLIRAffineUtils
  MLIRArithUtils
  MLIRDialectUtils
  MLIRGPUDialect
  MLIRXeVMDialect
  MLIRIR
  MLIRViewLikeInterface
  MLIRVectorDialect
)

# XeGPUOps.cpp transitively includes VersionTuple.h whose Minor/Subminor
# members conflict with a function-like macro defined earlier in the unity TU.
# set_source_files_properties doesn't propagate through llvm_add_library's
# internal OBJECT library, so disable unity on the OBJECT target directly.
if(TARGET obj.MLIRXeGPUDialect)
  set_target_properties(obj.MLIRXeGPUDialect PROPERTIES UNITY_BUILD OFF)
endif()
