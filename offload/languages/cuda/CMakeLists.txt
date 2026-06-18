add_llvm_library(
  LLVMcudart SHARED

  src/cuda_runtime.cpp

  LINK_COMPONENTS
  Support
  Offload
  OffloadKernel
  )

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(LLVMcudart PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_include_directories(LLVMcudart PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include/cuda
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include/generated
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/common/include)

target_compile_options(LLVMcudart PRIVATE ${offload_compile_flags} -fdeclspec)
target_link_options(LLVMcudart PRIVATE ${offload_link_flags})

target_compile_definitions(LLVMcudart PRIVATE
  TARGET_NAME="libLLVMcudart"
  DEBUG_PREFIX="LLVMcudart"
)

set_target_properties(LLVMcudart PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}/${OFFLOAD_TARGET_SUBDIR}"
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS LLVMcudart ARCHIVE DESTINATION "${OFFLOAD_INSTALL_LIBDIR}"
        COMPONENT offload)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/cuda/cuda_runtime.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload/cuda/)
