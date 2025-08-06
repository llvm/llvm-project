set(cuda_math_flags "")
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  file(GLOB libdevice_paths "${CUDAToolkit_LIBRARY_ROOT}/nvvm/libdevice/libdevice.*.bc")
  list(GET libdevice_paths 0 libdevice_path)

  if (EXISTS ${libdevice_path})
      list(APPEND cuda_math_flags "-Xclang" "-mlink-builtin-bitcode" "-Xclang" "${libdevice_path}")
      list(APPEND cuda_math_flags "-DCUDA_MATH_FOUND=1")
    endif()
endif()

set(hip_math_flags "")
find_package(AMDDeviceLibs QUIET HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(AMDDeviceLibs_FOUND)
  get_target_property(ocml_path ocml IMPORTED_LOCATION)
  list(APPEND hip_math_flags "-Xclang" "-mlink-builtin-bitcode" "-Xclang" "${ocml_path}")
  list(APPEND hip_math_flags "-DHIP_MATH_FOUND=1")
endif()

add_offload_test_device_code(CUDAMath.cpp cuda-math -O3 -stdlib -fno-builtin ${cuda_math_flags})
add_offload_test_device_code(HIPMath.cpp hip-math -O3 -stdlib -fno-builtin ${hip_math_flags})
add_offload_test_device_code(LLVMLibm.cpp llvm-libm -O3 -stdlib -fno-builtin)

add_custom_target(conformance_device_binaries DEPENDS
  cuda-math.bin
  hip-math.bin
  llvm-libm.bin
)
set(OFFLOAD_CONFORMANCE_DEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)
