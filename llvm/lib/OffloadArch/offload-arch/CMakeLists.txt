add_llvm_tool(offload-arch
  ${CMAKE_CURRENT_SOURCE_DIR}/offload-arch.cpp
  DEPENDS generated-table LLVMOffloadArch
)
target_link_libraries(offload-arch PRIVATE LLVMOffloadArch)

add_custom_target(
   amdgpu-offload-arch ALL
   COMMAND ln -sf offload-arch amdgpu-offload-arch
   DEPENDS offload-arch)
add_custom_target(
   nvidia-arch ALL
   COMMAND ln -sf offload-arch nvidia-arch
   DEPENDS offload-arch)

install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/amdgpu-offload-arch
    ${CMAKE_CURRENT_BINARY_DIR}/nvidia-arch
     DESTINATION bin
     COMPONENT offload-arch)
