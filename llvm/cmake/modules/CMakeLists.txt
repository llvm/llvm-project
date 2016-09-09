set(LLVM_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm)
set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")

# First for users who use an installed LLVM, create the
# LLVMInstallationExports.cmake
set(LLVM_EXPORTS_FILE ${llvm_cmake_builddir}/LLVMExports.cmake)
get_property(LLVM_EXPORTS GLOBAL PROPERTY LLVM_EXPORTS)
export(TARGETS ${LLVM_EXPORTS} FILE ${LLVM_EXPORTS_FILE})

# Then for users who want to link against the LLVM build tree, provide the
# normal targets and the build tree only targets.
set(LLVM_BUILDTREEONLY_EXPORTS_FILE ${llvm_cmake_builddir}/LLVMBuildTreeOnlyTargets.cmake)
get_property(LLVM_EXPORTS_BUILDTREE_ONLY GLOBAL PROPERTY LLVM_EXPORTS_BUILDTREE_ONLY)
export(TARGETS ${LLVM_EXPORTS_BUILDTREE_ONLY} FILE ${LLVM_BUILDTREEONLY_EXPORTS_FILE})

get_property(LLVM_AVAILABLE_LIBS GLOBAL PROPERTY LLVM_LIBS)

foreach(lib ${LLVM_AVAILABLE_LIBS})
  get_property(llvm_lib_deps GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${lib})
  set(all_llvm_lib_deps
    "${all_llvm_lib_deps}\nset_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_${lib} ${llvm_lib_deps})")
endforeach(lib)

# Generate LLVMConfig.cmake for the build tree.
set(LLVM_CONFIG_CODE "
# LLVM_BUILD_* values available only from LLVM build tree.
set(LLVM_BUILD_BINARY_DIR \"${LLVM_BINARY_DIR}\")
set(LLVM_BUILD_LIBRARY_DIR \"${LLVM_LIBRARY_DIR}\")
set(LLVM_BUILD_MAIN_INCLUDE_DIR \"${LLVM_MAIN_INCLUDE_DIR}\")
set(LLVM_BUILD_MAIN_SRC_DIR \"${LLVM_MAIN_SRC_DIR}\")
")
set(LLVM_CONFIG_INCLUDE_DIRS
  "${LLVM_MAIN_INCLUDE_DIR}"
  "${LLVM_INCLUDE_DIR}"
  )
set(LLVM_CONFIG_LIBRARY_DIRS
  "${LLVM_LIBRARY_DIR}"
  )
set(LLVM_CONFIG_CMAKE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(LLVM_CONFIG_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_CONFIG_TOOLS_BINARY_DIR "${LLVM_TOOLS_BINARY_DIR}")
# We need to use the full path to the LLVM Exports file to make sure we get the
# one from the build tree. This is due to our cmake files being split between
# this source dir and the binary dir in the build tree configuration and the
# LLVM_CONFIG_CMAKE_DIR being the source directory. In contrast in the install
# tree, both the generated LLVMExports.cmake file and the rest of the cmake
# source files are put in the same cmake directory.
set(LLVM_CONFIG_EXPORTS_FILE "${LLVM_EXPORTS_FILE}")
set(LLVM_CONFIG_EXPORTS "${LLVM_EXPORTS};${LLVM_EXPORTS_BUILDTREE_ONLY}")
set(llvm_config_include_buildtree_only_exports
"include(\"${LLVM_BUILDTREEONLY_EXPORTS_FILE}\")")
configure_file(
  LLVMConfig.cmake.in
  ${llvm_cmake_builddir}/LLVMConfig.cmake
  @ONLY)
set(llvm_config_include_buildtree_only_exports)

# For compatibility with projects that include(LLVMConfig)
# via CMAKE_MODULE_PATH, place API modules next to it.
# This should be removed in the future.
file(COPY .
  DESTINATION ${llvm_cmake_builddir}
  FILES_MATCHING PATTERN *.cmake
  PATTERN .svn EXCLUDE
  PATTERN CMakeFiles EXCLUDE
  )

# Generate LLVMConfig.cmake for the install tree.
set(LLVM_CONFIG_CODE "
# Compute the installation prefix from this LLVMConfig.cmake file location.
get_filename_component(LLVM_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)")
# Construct the proper number of get_filename_component(... PATH)
# calls to compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${LLVM_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(LLVM_CONFIG_CODE "${LLVM_CONFIG_CODE}
get_filename_component(LLVM_INSTALL_PREFIX \"\${LLVM_INSTALL_PREFIX}\" PATH)")
endforeach(p)
set(LLVM_CONFIG_INCLUDE_DIRS "\${LLVM_INSTALL_PREFIX}/include")
set(LLVM_CONFIG_LIBRARY_DIRS "\${LLVM_INSTALL_PREFIX}/lib\${LLVM_LIBDIR_SUFFIX}")
set(LLVM_CONFIG_CMAKE_DIR "\${LLVM_INSTALL_PREFIX}/${LLVM_INSTALL_PACKAGE_DIR}")
set(LLVM_CONFIG_BINARY_DIR "\${LLVM_INSTALL_PREFIX}")
set(LLVM_CONFIG_TOOLS_BINARY_DIR "\${LLVM_INSTALL_PREFIX}/bin")
set(LLVM_CONFIG_EXPORTS_FILE "\${LLVM_CMAKE_DIR}/LLVMExports.cmake")
set(LLVM_CONFIG_EXPORTS "${LLVM_EXPORTS}")
configure_file(
  LLVMConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LLVMConfig.cmake
  @ONLY)

# Generate LLVMConfigVersion.cmake for build and install tree.
configure_file(
  LLVMConfigVersion.cmake.in
  ${llvm_cmake_builddir}/LLVMConfigVersion.cmake
  @ONLY)

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(EXPORT LLVMExports DESTINATION ${LLVM_INSTALL_PACKAGE_DIR})

  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/LLVMConfig.cmake
    ${llvm_cmake_builddir}/LLVMConfigVersion.cmake
    LLVM-Config.cmake
    DESTINATION ${LLVM_INSTALL_PACKAGE_DIR})

  install(DIRECTORY .
    DESTINATION ${LLVM_INSTALL_PACKAGE_DIR}
    FILES_MATCHING PATTERN *.cmake
    PATTERN .svn EXCLUDE
    PATTERN LLVMConfig.cmake EXCLUDE
    PATTERN LLVMConfigVersion.cmake EXCLUDE
    PATTERN LLVM-Config.cmake EXCLUDE
    PATTERN GetHostTriple.cmake EXCLUDE
    PATTERN CheckAtomic.cmake EXCLUDE)
endif()
