# Keep this in sync with llvm/cmake/CMakeLists.txt!

include(GNUInstallPackageDir)
include(ExtendPath)
include(FindPrefixFromConfig)

set(POLLY_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/polly" CACHE STRING
  "Path for CMake subdirectory for Polly (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/polly')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
set(polly_cmake_builddir "${POLLY_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}/cmake/polly")

set(LLVM_INSTALL_PACKAGE_DIR "${CMAKE_INSTALL_PACKAGEDIR}/llvm" CACHE STRING
  "Path for CMake subdirectory for LLVM (defaults to '${CMAKE_INSTALL_PACKAGEDIR}/llvm')")
# CMAKE_INSTALL_PACKAGEDIR might be absolute, so don't reuse below.
string(REPLACE "${CMAKE_CFG_INTDIR}" "." llvm_cmake_builddir "${LLVM_LIBRARY_DIR}")
set(llvm_cmake_builddir "${llvm_cmake_builddir}/cmake/llvm")

if (CMAKE_CONFIGURATION_TYPES)
  set(POLLY_EXPORTS_FILE_NAME "PollyExports-$<LOWER_CASE:$<CONFIG>>.cmake")
else()
  # avoid conflicts in the build-tree when changing configuration
  set(POLLY_EXPORTS_FILE_NAME "PollyExports-all.cmake")
endif()

set(POLLY_CONFIG_EXPORTED_TARGETS Polly ${ISL_TARGET})
if (NOT WIN32 AND LLVM_ENABLE_PIC)
  # LLVMPolly is a dummy target on Win or if PIC code is disabled.
  list(APPEND POLLY_CONFIG_EXPORTED_TARGETS LLVMPolly)
endif()

# Get the target type for every exported target
foreach(tgt IN LISTS POLLY_CONFIG_EXPORTED_TARGETS)
  get_target_property(tgt_type ${tgt} TYPE)
  string(REPLACE "_LIBRARY" "" tgt_type ${tgt_type})
  set(POLLY_CONFIG_TARGET_${tgt}_TYPE ${tgt_type})
endforeach()

# generate the import code for bundled/undbundled libisl versions
if (NOT POLLY_BUNDLED_ISL)
  get_property(incl TARGET ISL PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  get_property(lib TARGET ISL PROPERTY INTERFACE_LINK_LIBRARIES)
  get_property(opt TARGET ISL PROPERTY INTERFACE_COMPILE_OPTIONS)
  set(ISL_CONFIG_CODE "
add_library(ISL INTERFACE IMPORTED)
set_property(TARGET ISL APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${incl})
set_property(TARGET ISL APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${lib})
set_property(TARGET ISL APPEND PROPERTY INTERFACE_COMPILE_OPTIONS ${opt})")
else()
  set(ISL_CONFIG_CODE "
if (NOT TARGET PollyISL)
  add_library(PollyISL ${POLLY_CONFIG_TARGET_PollyISL_TYPE} IMPORTED)
endif()")
endif()

# Generate PollyConfig.cmake for the build tree.
set(POLLY_CONFIG_CODE "")
set(POLLY_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")
set(POLLY_CONFIG_CMAKE_DIR "${polly_cmake_builddir}")
set(POLLY_CONFIG_INCLUDE_DIRS
  ${POLLY_SOURCE_DIR}/include
  ${ISL_INCLUDE_DIRS}
  ${POLLY_BINARY_DIR}/include
  )
set(POLLY_CONFIG_LIBRARY_DIRS "${POLLY_BINARY_DIR}/lib")

# set locations for imported targets
foreach(tgt IN LISTS POLLY_CONFIG_EXPORTED_TARGETS)
  get_target_property(tgt_type ${tgt} TYPE)
  if (NOT tgt_type STREQUAL "INTERFACE_LIBRARY")
    set(POLLY_EXPORTS
      "set_target_properties(${tgt} PROPERTIES
              IMPORTED_LOCATION$<$<NOT:$<CONFIG:>>:_$<UPPER_CASE:$<CONFIG>>> $<TARGET_FILE:${tgt}>)
${POLLY_EXPORTS}")
  endif()
endforeach(tgt)

# PollyConfig holds the target definitions and general settings, PollyExports
# the imported locations
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PollyConfig.cmake.in
  ${polly_cmake_builddir}/PollyConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PollyConfigVersion.cmake.in
  ${polly_cmake_builddir}/PollyConfigVersion.cmake
  @ONLY)

file(GENERATE
  OUTPUT ${polly_cmake_builddir}/${POLLY_EXPORTS_FILE_NAME}
  CONTENT "${POLLY_EXPORTS}")


# Generate PollyConfig.cmake for the install tree.
unset(POLLY_EXPORTS)
find_prefix_from_config(POLLY_CONFIG_CODE POLLY_INSTALL_PREFIX "${POLLY_INSTALL_PACKAGE_DIR}")
extend_path(POLLY_CONFIG_LLVM_CMAKE_DIR "\${POLLY_INSTALL_PREFIX}" "${LLVM_INSTALL_PACKAGE_DIR}")
extend_path(POLLY_CONFIG_CMAKE_DIR "\${POLLY_INSTALL_PREFIX}" "${POLLY_INSTALL_PACKAGE_DIR}")
extend_path(POLLY_CONFIG_LIBRARY_DIRS "\${POLLY_INSTALL_PREFIX}" "lib${LLVM_LIBDIR_SUFFIX}")
extend_path(base_includedir "\${POLLY_INSTALL_PREFIX}" "${CMAKE_INSTALL_INCLUDEDIR}")
if (POLLY_BUNDLED_ISL)
  set(POLLY_CONFIG_INCLUDE_DIRS
    "${base_includedir}"
    "${base_includedir}/polly"
    )
else()
  set(POLLY_CONFIG_INCLUDE_DIRS
    "${base_includedir}"
    ${ISL_INCLUDE_DIRS}
    )
endif()

# Set locations for imported targets. The path is constructed to be relative to
# the config file.
#
# Include the config code of PollyConfig.cmake to define POLLY_INSTALL_PREFIX
# anew in case the export file is included alone rather than from that file.
set(POLLY_EXPORTS
"${POLLY_CONFIG_CODE}

")
foreach(tgt IN LISTS POLLY_CONFIG_EXPORTED_TARGETS)
  get_target_property(tgt_type ${tgt} TYPE)
  if (tgt_type STREQUAL "EXECUTABLE")
    set(tgt_prefix "${CMAKE_INSTALL_BINDIR}")
  else()
    set(tgt_prefix "${CMAKE_INSTALL_LIBDIR}")
  endif()

  extend_path(tgt_path "\${POLLY_INSTALL_PREFIX}" "${tgt_prefix}/$<TARGET_FILE_NAME:${tgt}>")

  if (NOT tgt_type STREQUAL "INTERFACE_LIBRARY")
    set(POLLY_EXPORTS
"set_target_properties(${tgt} PROPERTIES
        IMPORTED_LOCATION$<$<NOT:$<CONFIG:>>:_$<UPPER_CASE:$<CONFIG>>> \"${tgt_path}\")
${POLLY_EXPORTS}")
  endif()
endforeach(tgt)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PollyConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PollyConfig.cmake
  @ONLY)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PollyConfigVersion.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PollyConfigVersion.cmake
  @ONLY)
file(GENERATE OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${POLLY_EXPORTS_FILE_NAME}
  CONTENT "${POLLY_EXPORTS}")

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PollyConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PollyConfigVersion.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${POLLY_EXPORTS_FILE_NAME}"
    DESTINATION "${POLLY_INSTALL_PACKAGE_DIR}")
endif ()
