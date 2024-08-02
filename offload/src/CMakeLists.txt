message(STATUS "Building offloading runtime library libomptarget.")

if(LIBOMP_STANDALONE)
  set(LIBOMP ${LIBOMP_STANDALONE})
else()
  set(LIBOMP omp)
endif()

add_llvm_library(omptarget
  SHARED

  device.cpp
  interface.cpp
  omptarget.cpp
  OmptTracing.cpp
  OmptTracingBuffer.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp


  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  Object

  LINK_LIBS
  PUBLIC
  ${LIBOMP}

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Don't override an externally defined RPATH
if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set_target_properties(omptarget PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib" BUILD_RPATH "$ORIGIN")
else()
  set_target_properties(omptarget PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_RPATH} BUILD_RPATH ${CMAKE_INSTALL_RPATH})
endif()
if (OPENMP_ENABLE_LIBOMPTARGET_PROFILING)
  # Add LLVMSupport dependency if profiling is enabled.
  # Linking with LLVM component libraries also requires
  # aligning the compile flags.
  llvm_update_compile_flags(omptarget)
  target_compile_definitions(omptarget PUBLIC OMPTARGET_PROFILE_ENABLED)
  target_link_libraries(omptarget PRIVATE LLVMSupport)
endif()

target_include_directories(omptarget PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

target_link_libraries(
  omptarget
  PRIVATE
  ${CMAKE_DL_LIBS}
  "-Wl,--no-allow-shlib-undefined")
# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(omptarget PRIVATE
  TARGET_NAME=omptarget
  DEBUG_PREFIX="omptarget"
)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
  target_link_libraries(omptarget PRIVATE omptarget.rtl.${plugin})
endforeach()

target_compile_options(omptarget PRIVATE ${offload_compile_flags})
target_link_options(omptarget PRIVATE ${offload_link_flags})

macro(check_plugin_target target)
if (TARGET omptarget.rtl.${target})
	list(APPEND LIBOMPTARGET_PLUGINS_TO_LOAD ${target})
endif()
endmacro()

set(LIBOMPTARGET_PLUGINS_TO_LOAD "" CACHE STRING
  "Comma separated list of plugin names to look for at runtime")
if (NOT LIBOMPTARGET_PLUGINS_TO_LOAD)
	check_plugin_target(amdgpu)
	check_plugin_target(host)
	check_plugin_target(cuda)
endif()

list(TRANSFORM LIBOMPTARGET_PLUGINS_TO_LOAD PREPEND "\"libomptarget.rtl.")
list(TRANSFORM LIBOMPTARGET_PLUGINS_TO_LOAD APPEND "\"")
list(JOIN LIBOMPTARGET_PLUGINS_TO_LOAD "," ENABLED_OFFLOAD_PLUGINS)
target_compile_definitions(omptarget PRIVATE ENABLED_OFFLOAD_PLUGINS=${ENABLED_OFFLOAD_PLUGINS})

if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set_target_properties(omptarget PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()


# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS omptarget LIBRARY COMPONENT omptarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
