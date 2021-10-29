if(CMAKE_HOST_UNIX)
add_llvm_tool(offload-arch
  ${CMAKE_CURRENT_SOURCE_DIR}/offload-arch.cpp
  DEPENDS generated-table LLVMOffloadArch
)
target_link_libraries(offload-arch PRIVATE LLVMOffloadArch)

if(CMAKE_HOST_UNIX)
  set(COMPILER_LINK_OR_COPY create_symlink)
  set(OFFARCH_NAME offload-arch)
else()
  set(COMPILER_LINK_OR_COPY copy)
  set(OFFARCH_NAME bin\\offload-arch.exe)
  set(WINPREFIX bin\\)
endif()

add_custom_target(
   amdgpu-offload-arch ALL
   COMMAND ${CMAKE_COMMAND} -E ${COMPILER_LINK_OR_COPY} ${OFFARCH_NAME} "${WINPREFIX}amdgpu-offload-arch"
   DEPENDS offload-arch)
add_custom_target(
   nvidia-arch ALL
   COMMAND ${CMAKE_COMMAND} -E ${COMPILER_LINK_OR_COPY} ${OFFARCH_NAME} "${WINPREFIX}nvidia-arch"
   DEPENDS offload-arch)

install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/amdgpu-offload-arch
    ${CMAKE_CURRENT_BINARY_DIR}/nvidia-arch
     DESTINATION bin
     COMPONENT offload-arch)
endif()
