# Keep this in sync with llvm/cmake/CMakeLists.txt!
if (LLVM_INSTALL_ROOT)
  # this simplifies things down the road, by not requiring to distinguish
  # between in-tree and out of tree builds
  set(LLVM_BINARY_DIR ${LLVM_INSTALL_ROOT})
endif()

set(LLVM_INSTALL_PACKAGE_DIR "lib${LLVM_LIBDIR_SUFFIX}/cmake/llvm")
set(POLLY_INSTALL_PACKAGE_DIR "lib${LLVM_LIBDIR_SUFFIX}/cmake/polly")
if (CMAKE_CONFIGURATION_TYPES)
  set(POLLY_EXPORTS_FILE_NAME "PollyExports-$<LOWER_CASE:$<CONFIG>>.cmake")
else()
  # avoid conflicts in the build-tree when changing configuration
  set(POLLY_EXPORTS_FILE_NAME "PollyExports-all.cmake")
endif()

set(POLLY_CONFIG_EXPORTED_TARGETS Polly ${ISL_TARGET})
if (NOT MSVC)
  # LLVMPolly is a dummy target on Win
  list(APPEND POLLY_CONFIG_EXPORTED_TARGETS LLVMPolly)
endif()
if (POLLY_ENABLE_GPGPU_CODEGEN)
  list(APPEND POLLY_CONFIG_EXPORTED_TARGETS PollyPPCG)
endif()

# Get the target type for every exported target
foreach(tgt IN LISTS POLLY_CONFIG_EXPORTED_TARGETS)
  get_target_property(tgt_type ${tgt} TYPE)
  string(REPLACE "_LIBRARY" "" tgt_type ${tgt_type})
  set(POLLY_CONFIG_TARGET_${tgt}_TYPE ${tgt_type})
endforeach()

set(llvm_cmake_builddir "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")
set(POLLY_CONFIG_LLVM_CMAKE_DIR "${llvm_cmake_builddir}")

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
set(POLLY_CONFIG_CMAKE_DIR "${CMAKE_BINARY_DIR}/${POLLY_INSTALL_PACKAGE_DIR}")
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
              IMPORTED_LOCATION_$<UPPER_CASE:$<CONFIG>> $<TARGET_FILE:${tgt}>)
${POLLY_EXPORTS}")
  endif()
endforeach(tgt)

# PollyConfig holds the target definitions and general settings, PollyExports
# the imported locations
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PollyConfig.cmake.in
  ${POLLY_CONFIG_CMAKE_DIR}/PollyConfig.cmake
  @ONLY)

file(GENERATE
  OUTPUT ${POLLY_CONFIG_CMAKE_DIR}/${POLLY_EXPORTS_FILE_NAME}
  CONTENT "${POLLY_EXPORTS}")


# Generate PollyConfig.cmake for the install tree.
unset(POLLY_EXPORTS)
set(POLLY_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(POLLY_CONFIG_LLVM_CMAKE_DIR "${LLVM_BINARY_DIR}/${LLVM_INSTALL_PACKAGE_DIR}")
set(POLLY_CONFIG_CMAKE_DIR "${POLLY_INSTALL_PREFIX}/${POLLY_INSTALL_PACKAGE_DIR}")
set(POLLY_CONFIG_LIBRARY_DIRS "${POLLY_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX}")
if (POLLY_BUNDLED_ISL)
  set(POLLY_CONFIG_INCLUDE_DIRS
    "${POLLY_INSTALL_PREFIX}/include"
    "${POLLY_INSTALL_PREFIX}/include/polly"
    )
else()
  set(POLLY_CONFIG_INCLUDE_DIRS
    "${POLLY_INSTALL_PREFIX}/include"
    ${ISL_INCLUDE_DIRS}
    )
endif()

# set locations for imported targets. The path is constructed to be relative to
# the config file
foreach(tgt IN LISTS POLLY_CONFIG_EXPORTED_TARGETS)
  get_target_property(tgt_type ${tgt} TYPE)
  if (tgt_type STREQUAL "EXECUTABLE")
    set(tgt_prefix "bin/")
  else()
    set(tgt_prefix "lib/")
  endif()

  set(tgt_path "${CMAKE_INSTALL_PREFIX}/${tgt_prefix}$<TARGET_FILE_NAME:${tgt}>")
  file(RELATIVE_PATH tgt_path ${POLLY_CONFIG_CMAKE_DIR} ${tgt_path})

  if (NOT tgt_type STREQUAL "INTERFACE_LIBRARY")
    set(POLLY_EXPORTS
"set_target_properties(${tgt} PROPERTIES
        IMPORTED_LOCATION$<$<NOT:$<CONFIG:>>:_<UPPER_CASE:$<CONFIG>> \${CMAKE_CURRENT_LIST_DIR}/${tgt_path})
${POLLY_EXPORTS}")
  endif()
endforeach(tgt)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/PollyConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PollyConfig.cmake
  @ONLY)
file(GENERATE OUTPUT
  ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${POLLY_EXPORTS_FILE_NAME}
  CONTENT "${POLLY_EXPORTS}")

if (NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    FILES
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/PollyConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${POLLY_EXPORTS_FILE_NAME}"
    DESTINATION "${POLLY_INSTALL_PACKAGE_DIR}")
endif ()
