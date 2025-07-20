add_custom_target(offload.conformance)

set(PLUGINS_TEST_COMMON MathTest)

set(OFFLOAD_CONFORMANCE_PROVIDER_PLATFORMS_llvm-libm "cuda;amdgpu")
set(OFFLOAD_CONFORMANCE_PROVIDER_PLATFORMS_cuda-math "cuda")
set(OFFLOAD_CONFORMANCE_PROVIDER_PLATFORMS_hip-math "amdgpu")

set(OFFLOAD_CONFORMANCE_AVAILABLE_PLATFORMS "")

if("cuda" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
  find_package(CUDAToolkit QUIET)
  if(PLATFORM_HAS_NVPTX AND CUDAToolkit_FOUND)
    list(APPEND OFFLOAD_CONFORMANCE_AVAILABLE_PLATFORMS "cuda")
  endif()
endif()
if("amdgpu" IN_LIST LIBOMPTARGET_PLUGINS_TO_BUILD)
  if(PLATFORM_HAS_AMDGPU)
    list(APPEND OFFLOAD_CONFORMANCE_AVAILABLE_PLATFORMS "amdgpu")
  endif()
endif()

add_subdirectory(device_code)
add_subdirectory(lib)
add_subdirectory(tests)
