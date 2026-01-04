set(LLVM_LINK_COMPONENTS
  BinaryFormat
  Object
  Option
  Support
  FrontendOffloading
  TargetParser
)

add_llvm_tool(llvm-gpu-loader
  llvm-gpu-loader.cpp

  # TODO: We intentionally split this currently due to statically linking the
  #       GPU runtimes. Dynamically load the dependencies, possibly using the
  #       LLVM offloading API when it is complete.
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen
)

# Locate the RPC server handling interface.
include(FindLibcCommonUtils)
target_link_libraries(llvm-gpu-loader PUBLIC llvm-libc-common-utilities)

# Check for HSA support for targeting AMD GPUs.
find_package(hsa-runtime64 QUIET 1.2.0 HINTS ${CMAKE_INSTALL_PREFIX} PATHS /opt/rocm)
if(hsa-runtime64_FOUND)
  target_sources(llvm-gpu-loader PRIVATE amdhsa.cpp)
  target_compile_definitions(llvm-gpu-loader PRIVATE AMDHSA_SUPPORT)
  target_link_libraries(llvm-gpu-loader PRIVATE hsa-runtime64::hsa-runtime64)

  # Compatibility with the old amdhsa-loader name.
  add_llvm_tool_symlink(amdhsa-loader llvm-gpu-loader)
endif()

# Check for CUDA support for targeting NVIDIA GPUs.
find_package(CUDAToolkit 11.2 QUIET)
if(CUDAToolkit_FOUND)
  target_sources(llvm-gpu-loader PRIVATE nvptx.cpp)
  target_compile_definitions(llvm-gpu-loader PRIVATE NVPTX_SUPPORT)
  target_link_libraries(llvm-gpu-loader PRIVATE CUDA::cuda_driver)

  # Compatibility with the old nvptx-loader name.
  add_llvm_tool_symlink(nvptx-loader llvm-gpu-loader)
endif()
