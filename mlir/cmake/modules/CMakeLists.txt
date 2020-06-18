# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(MLIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir)
set(mlir_cmake_builddir "${CMAKE_BINARY_DIR}/${MLIR_INSTALL_PACKAGE_DIR}")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

get_property(MLIR_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)
export(TARGETS ${MLIR_EXPORTS} FILE ${mlir_cmake_builddir}/MLIRTargets.cmake)

get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# Generate MlirConfig.cmake for the build tree.
set(MLIR_CONFIG_CMAKE_DIR "${mlir_cmake_builddir}")
set(MLIR_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(MLIR_CONFIG_EXPORTS_FILE "\${MLIR_CMAKE_DIR}/MLIRTargets.cmake")
set(MLIR_CONFIG_INCLUDE_DIRS
  "${MLIR_SOURCE_DIR}/include"
  "${MLIR_BINARY_DIR}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfig.cmake.in
  ${mlir_cmake_builddir}/MLIRConfig.cmake
  @ONLY)
set(MLIR_CONFIG_CMAKE_DIR)
set(MLIR_CONFIG_LLVM_CMAKE_DIR)
set(MLIR_CONFIG_EXPORTS_FILE)
set(MLIR_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that include(MLIRConfig)
# via CMAKE_MODULE_PATH, place API modules next to it.
# This should be removed in the future.
file(COPY .
  DESTINATION ${mlir_cmake_builddir}
  FILES_MATCHING PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
  )

# Generate MLIRConfig.cmake for the install tree.
set(MLIR_CONFIG_CODE "
# Compute the installation prefix from this MLIRConfig.cmake file location.
get_filename_component(MLIR_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${MLIR_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(MLIR_CONFIG_CODE "${MLIR_CONFIG_CODE}
get_filename_component(MLIR_INSTALL_PREFIX \"\${MLIR_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(MLIR_CONFIG_CMAKE_DIR "\${MLIR_INSTALL_PREFIX}/${MLIR_INSTALL_PACKAGE_DIR}")
set(MLIR_CONFIG_LLVM_CMAKE_DIR "\${MLIR_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(MLIR_CONFIG_EXPORTS_FILE "\${MLIR_CMAKE_DIR}/MLIRTargets.cmake")
set(MLIR_CONFIG_INCLUDE_DIRS
  "\${MLIR_INSTALL_PREFIX}/include"
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MLIRConfig.cmake
  @ONLY)
set(MLIR_CONFIG_CODE)
set(MLIR_CONFIG_CMAKE_DIR)
set(MLIR_CONFIG_LLVM_CMAKE_DIR)
set(MLIR_CONFIG_EXPORTS_FILE)
set(MLIR_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the MLIR parts as well
  # Include the cmake files so other tools can use mlir-tblgen, etc.
  get_property(mlir_has_exports GLOBAL PROPERTY MLIR_HAS_EXPORTS)
  if(mlir_has_exports)
    install(EXPORT MLIRTargets DESTINATION ${MLIR_INSTALL_PACKAGE_DIR}
            COMPONENT mlir-cmake-exports)
  endif()

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MLIRConfig.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddMLIR.cmake
    DESTINATION ${MLIR_INSTALL_PACKAGE_DIR}
    COMPONENT mlir-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(mlir-cmake-exports)
    add_llvm_install_targets(install-mlir-cmake-exports
                             COMPONENT mlir-cmake-exports)
  endif()
endif()
