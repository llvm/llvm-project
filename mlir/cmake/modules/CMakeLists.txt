include(GNUInstallPackageDir)
include(ExtendPath)
include(LLVMDistributionSupport)
include(FindPrefixFromConfig)

# Generate a list of CMake library targets so that other CMake projects can
# link against them. LLVM calls its version of this file LLVMExports.cmake, but
# the usual CMake convention seems to be ${Project}Targets.cmake.
set(MLIR_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/mlir" CACHE STRING
  "Path for CMake subdirectory for Polly (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/polly')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
set(mlir_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/llvm" CACHE STRING
  "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
string(REPLACE "${CMAKE_CFG_INTDIR}" "." llvm_cmake_builddir "${LLVM_LIBRARY_DIR}")
set(llvm_cmake_builddir "${llvm_cmake_builddir}/cmake/llvm")

get_property(MLIR_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)
export(TARGETS ${MLIR_EXPORTS} FILE ${mlir_cmake_builddir}/MLIRTargets.cmake)

get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

# Generate MlirConfig.cmake for the build tree.
set(MLIR_CONFIG_CMAKE_DIR "${mlir_cmake_builddir}")
set(MLIR_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(MLIR_CONFIG_INCLUDE_EXPORTS "include(\"\${MLIR_CMAKE_DIR}/MLIRTargets.cmake\")")
set(MLIR_CONFIG_INCLUDE_DIRS
  "${MLIR_SOURCE_DIR}/include"
  "${MLIR_BINARY_DIR}/include"
  )
# Refer to the best host mlir-tbgen, which might be a host-optimized version
set(MLIR_CONFIG_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}")
set(MLIR_CONFIG_PDLL_TABLEGEN_EXE "${MLIR_PDLL_TABLEGEN_EXE}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfig.cmake.in
  ${mlir_cmake_builddir}/MLIRConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfigVersion.cmake.in
  ${mlir_cmake_builddir}/MLIRConfigVersion.cmake
  @ONLY)
set(MLIR_CONFIG_CMAKE_DIR)
set(MLIR_CONFIG_LLVM_CMAKE_DIR)
set(MLIR_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that include(MLIRConfig)
# via CMAKE_MODULE_PATH, place API modules next to it.
# Copy without source permissions because the source could be read-only,
# but we need to write into the copied folder.
# This should be removed in the future.
file(COPY .
  DESTINATION ${mlir_cmake_builddir}
  NO_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
  )

# Generate MLIRConfig.cmake for the install tree.
find_prefix_from_config(MLIR_CONFIG_CODE MLIR_INSTALL_PREFIX "${MLIR_INSTALL_PACKAGE_DIR}")
extend_path(MLIR_CONFIG_CMAKE_DIR "\${MLIR_INSTALL_PREFIX}" "${MLIR_INSTALL_PACKAGE_DIR}")
extend_path(MLIR_CONFIG_LLVM_CMAKE_DIR "\${MLIR_INSTALL_PREFIX}" "${LLVM_INSTALL_PACKAGE_DIR}")
get_config_exports_includes(MLIR MLIR_CONFIG_INCLUDE_EXPORTS)
extend_path(base_includedir "\${MLIR_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")
set(MLIR_CONFIG_INCLUDE_DIRS
  "${base_includedir}"
  )
# Ensure that we are using the installed mlir-tblgen. This might not be MLIR_TABLEGEN_EXE
# if we're building with a host-optimized mlir-tblgen (with LLVM_OPTIMIZED_TABLEGEN).
set(MLIR_CONFIG_TABLEGEN_EXE mlir-tblgen)
set(MLIR_CONFIG_PDLL_TABLEGEN_EXE mlir-pdll)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MLIRConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MLIRConfigVersion.cmake
  @ONLY)
set(MLIR_CONFIG_CODE)
set(MLIR_CONFIG_CMAKE_DIR)
set(MLIR_CONFIG_LLVM_CMAKE_DIR)
set(MLIR_CONFIG_INCLUDE_DIRS)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  # Not TOOLCHAIN ONLY, so install the MLIR parts as well
  # Include the cmake files so other tools can use mlir-tblgen, etc.
  install_distribution_exports(MLIR)

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MLIRConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/MLIRConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddMLIR.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/AddMLIRPython.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/MLIRDetectPythonEnv.cmake
    DESTINATION ${MLIR_INSTALL_PACKAGE_DIR}
    COMPONENT mlir-cmake-exports)

  if(NOT LLVM_ENABLE_IDE)
    # Add a dummy target so this can be used with LLVM_DISTRIBUTION_COMPONENTS
    add_custom_target(mlir-cmake-exports)
    add_llvm_install_targets(install-mlir-cmake-exports
                             COMPONENT mlir-cmake-exports)
  endif()
endif()
