# This library is shared by both MLIRSparseTensorDialect and
# MLIRSparseTensorRuntime, so it must not depend on any of the MLIR/LLVM
# internals or else mlir_c_runner_utils will inherit that dependency.
#
# Because this is a header-only (`INTERFACE`) library, we cannot use
# the `add_mlir_library` function.  So we do our best to replicate the
# relevant portions below.  If doing so becomes too complicated, then
# we should adjust the `add_mlir_library` function to also work for
# `INTERFACE` libraries.
set(MLIRSparseTensorEnums_hdrs
    mlir/Dialect/SparseTensor/IR/Enums.h)
if(MSVC_IDE OR XCODE)
  foreach(hdr ${MLIRSparseTensorEnums_hdrs})
    set_source_files_properties(${MLIR_MAIN_INCLUDE_DIR}/${hdr}
      PROPERTIES HEADER_FILE_ONLY ON)
  endforeach()
endif()
add_library(MLIRSparseTensorEnums INTERFACE)
foreach(hdr ${MLIRSparseTensorEnums_hdrs})
  target_sources(MLIRSparseTensorEnums INTERFACE
    $<BUILD_INTERFACE:${MLIR_MAIN_INCLUDE_DIR}/${hdr}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${hdr}>)
endforeach()
if(TARGET MLIRSparseTensorEnums)
  add_mlir_library_install(MLIRSparseTensorEnums)
else()
  add_custom_target(MLIRSparseTensorEnums) # empty "phony" target
endif()

add_mlir_dialect_library(MLIRSparseTensorDialect
  SparseTensorDialect.cpp
  SparseTensorInterfaces.cpp
  Detail/Var.cpp
  Detail/DimLvlMap.cpp
  Detail/LvlTypeParser.cpp
  Detail/DimLvlMapParser.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  MLIRSparseTensorAttrDefsIncGen
  MLIRSparseTensorOpsIncGen
  MLIRSparseTensorTypesIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialect
  MLIRDialectUtils
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport
  MLIRSparseTensorEnums
  )
