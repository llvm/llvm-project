if(DEFINED ROCM_PATH AND EXISTS ${ROCM_PATH})
  add_subdirectory(AMDGPU)
endif()
add_subdirectory(MockGPU)
