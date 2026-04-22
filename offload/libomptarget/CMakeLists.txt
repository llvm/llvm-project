message(STATUS "Building offloading runtime library libomptarget.")

if(NOT TARGET "omp")
  message(FATAL_ERROR "Error: omp target dependency of libomptarget not found. Please ensure 'openmp' is on LLVM_ENABLE_RUNTIMES list.")
endif()

add_library(omptarget SHARED
  device.cpp
  interface.cpp
  omptarget.cpp
  OffloadRTL.cpp
  LegacyAPI.cpp
  PluginManager.cpp
  DeviceImage.cpp

  OpenMP/API.cpp
  OpenMP/Mapping.cpp
  OpenMP/InteropAPI.cpp
  OpenMP/OMPT/Callback.cpp
  OpenMP/OMPT/OmptTracing.cpp
  OpenMP/OMPT/OmptTracingBuffer.cpp

  KernelLanguage/API.cpp
)

if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs FrontendOpenMP Support Object)
endif()
target_link_libraries(omptarget PRIVATE omp ${llvm_libs})
target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(omptarget PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

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
if (OMPT_TARGET_DEFAULT AND LIBOMPTARGET_OMPT_SUPPORT)
  # OmptProfiler.cpp provides the strong getProfilerToAttach() that routes
  # plugin device events through OmptProfilerTy. It must be compiled directly
  # into omptarget so its strong definition unconditionally overrides the weak
  # fallback in GenericProfiler.cpp (which is embedded into each plugin archive
  # via the PluginCommon OBJECT library and would otherwise win the link).
  target_sources(omptarget PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/OMPT/OmptProfiler.cpp
  )
  target_include_directories(omptarget PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/OMPT
    ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include
  )
  # PluginOmpt provides OmptTracing.cpp (plugin-side GPU tracing).
  target_link_libraries(omptarget PRIVATE PluginOmpt)
endif()

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
if(UNIX AND NOT APPLE)
  set_target_properties(omptarget PROPERTIES
                        SOVERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}"
                        VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}")
endif()
install(TARGETS omptarget LIBRARY COMPONENT offload DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
