# As of rocm-3.7, hsa is installed with cmake packages and kmt is found via hsa
find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.amdgpu AMDGPU)

target_sources(omptarget.rtl.amdgpu PRIVATE src/rtl.cpp)
target_include_directories(omptarget.rtl.amdgpu PRIVATE
                           ${CMAKE_CURRENT_SOURCE_DIR}/utils)

if(hsa-runtime64_FOUND AND NOT "amdgpu" IN_LIST LIBOMPTARGET_DLOPEN_PLUGINS)
  message(STATUS "Building AMDGPU plugin linked against libhsa")
  target_link_libraries(omptarget.rtl.amdgpu PRIVATE hsa-runtime64::hsa-runtime64)
else()
  message(STATUS "Building AMDGPU plugin for dlopened libhsa")
  target_include_directories(omptarget.rtl.amdgpu PRIVATE dynamic_hsa)
  target_sources(omptarget.rtl.amdgpu PRIVATE dynamic_hsa/hsa.cpp)
endif()

# Configure testing for the AMDGPU plugin. We will build tests if we could a
# functional AMD GPU on the system, or if manually specifies by the user.
option(LIBOMPTARGET_FORCE_AMDGPU_TESTS "Build AMDGPU libomptarget tests" OFF)
if (LIBOMPTARGET_FOUND_AMDGPU_GPU OR LIBOMPTARGET_FORCE_AMDGPU_TESTS)
  # Report to the parent scope that we are building a plugin for amdgpu
  set(LIBOMPTARGET_SYSTEM_TARGETS
      "${LIBOMPTARGET_SYSTEM_TARGETS} amdgcn-amd-amdhsa" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.amdgpu")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  message(STATUS "Not generating AMDGPU tests, no supported devices detected."
                   " Use 'LIBOMPTARGET_FORCE_AMDGPU_TESTS' to override.")
endif()
