# This file configures the CMake package configuration file "MLIRConfig.cmake"
# (see https://cmake.org/cmake/help/latest/manual/cmake-packages.7.html).
# Two different versions of the package config file are generated and copied
# into different locations:
# 1. One in a canonical location in the build directory,
#   '<build>/lib/cmake/mlir/MLIRConfig.cmake'. This version may use absolute
#    paths referring to locations in the build tree.
# 2. One under '${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles', which is the version
#    *to be installed* when `cmake --install` command is run or when CPack
#    is invoked. This is the version that is distributed with LLVM packages.
#    The key difference between this and (1) is that all paths must be
#    relative the the installed "MLIRConfig.cmake" file.

include(GNUInstallPackageDir)
include(ExtendPath)
include(LLVMDistributionSupport)
include(FindPrefixFromConfig)

# This is a relative path to the location of MLIRConfig.cmake within the
# installation.
set(MLIR_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/mlir" CACHE STRING
  "Path for CMake subdirectory for MLIR (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/mlir')")

# Keep this in sync with llvm/cmake/CMakeLists.txt!
set(LLVM_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/llvm" CACHE STRING
  "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')")
string(REPLACE "${CMAKE_CFG_INTDIR}" "." llvm_cmake_builddir "${LLVM_LIBRARY_DIR}")

# These two variables contain absolute paths, so only use them for the
# build directory 'MLIRConfig.cmake'.
set(mlir_cmake_builddir "${CMAKE_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/mlir")
set(llvm_cmake_builddir "${llvm_cmake_builddir}/cmake/llvm")

# -----------------------------------------------------------------------------
# Generate `<build>/lib/cmake/mlir/MLIRConfig.cmake'
# -----------------------------------------------------------------------------

# Generate the 'MLIRTarget.cmake' file.
get_property(MLIR_EXPORTS GLOBAL PROPERTY MLIR_EXPORTS)
export(TARGETS ${MLIR_EXPORTS} FILE ${mlir_cmake_builddir}/MLIRTargets.cmake)

# Declare all the variables required by the 'MLIRConfig.cmake.in' template.
get_property(MLIR_ALL_LIBS GLOBAL PROPERTY MLIR_ALL_LIBS)
get_property(MLIR_DIALECT_LIBS GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(MLIR_CONVERSION_LIBS GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(MLIR_EXTENSION_LIBS GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(MLIR_TRANSLATION_LIBS GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(MLIR_CAPI_LIBS GLOBAL PROPERTY MLIR_CAPI_LIBS)

# Generate MlirConfig.cmake for the build tree.
set(MLIR_CONFIG_CMAKE_DIR "${mlir_cmake_builddir}")
set(MLIR_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(MLIR_CONFIG_INCLUDE_EXPORTS "include(\"\${MLIR_CMAKE_DIR}/MLIRTargets.cmake\")")
set(MLIR_CONFIG_INCLUDE_DIRS
  "${MLIR_SOURCE_DIR}/include"
  "${MLIR_BINARY_DIR}/include"
  )
# Refer to the best host mlir-tblgen, which might be a host-optimized version
set(MLIR_CONFIG_TABLEGEN_EXE "${MLIR_TABLEGEN_EXE}")
set(MLIR_CONFIG_PDLL_TABLEGEN_EXE "${MLIR_PDLL_TABLEGEN_EXE}")
set(MLIR_CONFIG_SRC_SHARDER_TABLEGEN_EXE "${MLIR_SRC_SHARDER_TABLEGEN_EXE}")

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfig.cmake.in
  ${mlir_cmake_builddir}/MLIRConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/MLIRConfigVersion.cmake.in
  ${mlir_cmake_builddir}/MLIRConfigVersion.cmake
  @ONLY)

# Reset the variables which are invalid for the installed version of
# MLIRConfig.cmake.
set(MLIR_CONFIG_CMAKE_DIR)
set(MLIR_CONFIG_LLVM_CMAKE_DIR)
set(MLIR_CONFIG_INCLUDE_DIRS)

# For compatibility with projects that add '<build>/lib/cmake/mlir' to
# their CMAKE_MODULE_PATH, place API modules next to it.
# Copy without source permissions because the source could be read-only,
# but we need to write into the copied folder.
file(COPY .
  DESTINATION ${mlir_cmake_builddir}
  NO_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN *.cmake
  PATTERN CMakeFiles EXCLUDE
  )

# -----------------------------------------------------------------------------
# Generate MLIRConfig.cmake for the install tree.
# All path variables used in the template must be relative!
# -----------------------------------------------------------------------------

# This is a helper that sets 'MLIR_CONFIG_CODE' to some CMake code for
# insertion at the top of the generated 'MLIRConfig.cmake'. The code calculates
# the install directory prefix relative to the installed MLIRConfig.cmake (whose
# prefix is MLIR_INSTALL_PACKAGE_DIR), avoiding any use of absolute paths.
find_prefix_from_config(MLIR_CONFIG_CODE MLIR_INSTALL_PREFIX "${MLIR_INSTALL_PACKAGE_DIR}")

# Generate the other required paths by setting them relative to the
# variable 'MLIR_INSTALL_PREFIX'.
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
set(MLIR_CONFIG_SRC_SHARDER_TABLEGEN_EXE mlir-src-sharder)

set(MLIR_CONFIG_IRDL_TO_CPP_EXE mlir-irdl-to-cpp)

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
    ${CMAKE_CURRENT_SOURCE_DIR}/IRDLToCpp.cmake
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
