message(STATUS "Building OpenMP offloading runtime library libomptarget.")

if(NOT TARGET "omp")
  message(FATAL_ERROR "Error: omp target dependency of libomptarget not found. Please ensure 'openmp' is on LLVM_ENABLE_RUNTIMES list.")
endif()

add_llvm_library(omptarget
  SHARED

  interface.cpp
  omptarget.cpp
  LegacyAPI.cpp

  OpenMP/API.cpp
  OpenMP/InteropAPI.cpp

  KernelLanguage/API.cpp

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  Object

  LINK_LIBS
  PUBLIC
  omp

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)
target_include_directories(omptarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(omptarget PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
  set_property(TARGET omptarget APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports)
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(omptarget PRIVATE
  TARGET_NAME=omptarget
  DEBUG_PREFIX="omptarget"
)

target_link_libraries(omptarget PRIVATE ompaccsupport)

target_compile_options(omptarget PRIVATE ${offload_compile_flags})
target_link_options(omptarget PRIVATE ${offload_link_flags})

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(omptarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS omptarget LIBRARY COMPONENT offload DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
