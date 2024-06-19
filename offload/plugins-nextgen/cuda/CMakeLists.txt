# Create the library and add the default arguments.
add_target_library(omptarget.rtl.cuda CUDA)

target_sources(omptarget.rtl.cuda PRIVATE src/rtl.cpp)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND AND NOT "cuda" IN_LIST LIBOMPTARGET_DLOPEN_PLUGINS)
  message(STATUS "Building CUDA plugin linked against libcuda")
  target_link_libraries(omptarget.rtl.cuda PRIVATE CUDA::cuda_driver)
else()
  message(STATUS "Building CUDA plugin for dlopened libcuda")
  target_include_directories(omptarget.rtl.cuda PRIVATE dynamic_cuda)
  target_sources(omptarget.rtl.cuda PRIVATE dynamic_cuda/cuda.cpp)
endif()

# Configure testing for the CUDA plugin. We will build tests if we could a
# functional NVIDIA GPU on the system, or if manually specifies by the user.
option(LIBOMPTARGET_FORCE_NVIDIA_TESTS "Build NVIDIA libomptarget tests" OFF)
if (LIBOMPTARGET_FOUND_NVIDIA_GPU OR LIBOMPTARGET_FORCE_NVIDIA_TESTS)
  message(STATUS "Enable tests using CUDA plugin")
  set(LIBOMPTARGET_SYSTEM_TARGETS
      "${LIBOMPTARGET_SYSTEM_TARGETS} nvptx64-nvidia-cuda nvptx64-nvidia-cuda-LTO" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.cuda")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  message(STATUS "Not generating NVIDIA tests, no supported devices detected."
                   " Use 'LIBOMPTARGET_FORCE_NVIDIA_TESTS' to override.")
endif()
