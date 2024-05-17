if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building CUDA NextGen offloading plugin: only support CUDA in Linux x86_64, ppc64le, or aarch64 hosts.")
  return()
endif()

libomptarget_say("Building CUDA NextGen offloading plugin.")

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.cuda CUDA)

target_sources(omptarget.rtl.cuda PRIVATE src/rtl.cpp)

option(LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA "Build with dlopened libcuda" OFF)
if(LIBOMPTARGET_DEP_CUDA_FOUND AND NOT LIBOMPTARGET_FORCE_DLOPEN_LIBCUDA)
  libomptarget_say("Building CUDA plugin linked against libcuda")
  target_link_libraries(omptarget.rtl.cuda PRIVATE CUDA::cuda_driver)
else()
  libomptarget_say("Building CUDA plugin for dlopened libcuda")
  target_include_directories(omptarget.rtl.cuda PRIVATE dynamic_cuda)
  target_sources(omptarget.rtl.cuda PRIVATE dynamic_cuda/cuda.cpp)
endif()

# Configure testing for the CUDA plugin. We will build tests if we could a
# functional NVIDIA GPU on the system, or if manually specifies by the user.
option(LIBOMPTARGET_FORCE_NVIDIA_TESTS "Build NVIDIA libomptarget tests" OFF)
if (LIBOMPTARGET_FOUND_NVIDIA_GPU OR LIBOMPTARGET_FORCE_NVIDIA_TESTS)
  libomptarget_say("Enable tests using CUDA plugin")
  set(LIBOMPTARGET_SYSTEM_TARGETS
      "${LIBOMPTARGET_SYSTEM_TARGETS} nvptx64-nvidia-cuda nvptx64-nvidia-cuda-LTO" PARENT_SCOPE)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS "omptarget.rtl.cuda")
  set(LIBOMPTARGET_TESTED_PLUGINS "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  libomptarget_say("Not generating NVIDIA tests, no supported devices detected."
                   " Use 'LIBOMPTARGET_FORCE_NVIDIA_TESTS' to override.")
endif()
