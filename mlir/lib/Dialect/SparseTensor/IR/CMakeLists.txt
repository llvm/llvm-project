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
# This conditional is copypasta from `add_mlir_library`.
if(MSVC_IDE OR XCODE)
  foreach(hdr ${MLIRSparseTensorEnums_hdrs})
    set_source_files_properties(${MLIR_MAIN_INCLUDE_DIR}/${hdr}
      PROPERTIES HEADER_FILE_ONLY ON)
  endforeach()
endif()
# Older versions of cmake (< 3.19) require INTERFACE libraries to separate
# the `add_library` and `target_sources` calls.
add_library(MLIRSparseTensorEnums INTERFACE)
# If we call target_sources naively, then the library won't be
# installed properly (i.e., so that it can be used by projects outside
# the llvm-project repo).  To correct this, we must use BUILD_INTERFACE /
# INSTALL_INTERFACE generator expressions to avoid paths being interpreted
# as absolute; for more details, see <https://stackoverflow.com/a/62465051>.
# Unfortunately BUILD_INTERFACE and INSTALL_INTERFACE require their file
# paths to be relative to different things, hence why we use a foreach loop
# to iterate over the sources (in case we ever need to have more than one).
foreach(hdr ${MLIRSparseTensorEnums_hdrs})
  target_sources(MLIRSparseTensorEnums INTERFACE
    $<BUILD_INTERFACE:${MLIR_MAIN_INCLUDE_DIR}/${hdr}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/${hdr}>)
endforeach()
# The `add_mlir_library_install` is required for other libraries to
# depend on this one, but the conditional itself and the phony target
# are copypasta from `add_mlir_library`.  Afaict (wrengr), the version
# in `add_mlir_library` is to prevent `add_mlir_library_install` from
# raising additional errors whenever the underlying call to `add_llvm_library`
# fails.  However, since we are using `add_library` directly, I'm not
# sure if the conditional is helpful/required here or not.
if(TARGET MLIRSparseTensorEnums)
  add_mlir_library_install(MLIRSparseTensorEnums)
else()
  # Add empty "phony" target
  add_custom_target(MLIRSparseTensorEnums)
endif()
# Older versions of cmake (i.e., 3.18.0 but not 3.24.2) disallow setting
# the CXX_STANDARD property for INTERFACE libraries.  However, this library
# must adhere to the same CXX_STANDARD restriction as mlir_c_runner_utils.


add_mlir_dialect_library(MLIRSparseTensorDialect
  SparseTensorDialect.cpp
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
