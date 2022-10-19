# This library is shared by both MLIRSparseTensorDialect and
# MLIRSparseTensorRuntime, so it must not depend on any of the MLIR/LLVM
# internals or else mlir_c_runner_utils will inherit that dependency.
#
# Because this is a header-only (`INTERFACE`) library, we cannot use
# the `add_mlir_library` function.  So we do our best to replicate the
# relevant portions below.  If doing so becomes too complicated, then
# we should adjust the `add_mlir_library` function to also work for
# `INTERFACE` libraries.
set(MLIRSparseTensorEnums_srcs
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor/IR/Enums.h)
add_library(MLIRSparseTensorEnums INTERFACE ${MLIRSparseTensorEnums_srcs})
if(MSVC_IDE OR XCODE)
  set_source_files_properties(${MLIRSparseTensorEnums_srcs}
    PROPERTIES HEADER_FILE_ONLY ON)
endif()
add_mlir_library_install(MLIRSparseTensorEnums)
set_property(TARGET MLIRSparseTensorEnums PROPERTY CXX_STANDARD 17)


add_mlir_dialect_library(MLIRSparseTensorDialect
  SparseTensorDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/SparseTensor

  DEPENDS
  MLIRSparseTensorAttrDefsIncGen
  MLIRSparseTensorOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialect
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSupport
  MLIRSparseTensorEnums
  )
