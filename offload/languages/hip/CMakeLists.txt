add_llvm_library(
  LLVMhip64 SHARED

  src/hip_runtime.cpp

  LINK_COMPONENTS
  Support
  Offload
  OffloadKernel
  )

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(LLVMhip64 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_include_directories(LLVMhip64 PUBLIC
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include/hip
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include/generated
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/common/include)

target_compile_options(LLVMhip64 PRIVATE ${offload_compile_flags} -fdeclspec)
target_link_options(LLVMhip64 PRIVATE ${offload_link_flags})

target_compile_definitions(LLVMhip64 PRIVATE
  TARGET_NAME="libLLVMhiprt"
  DEBUG_PREFIX="LLVMhip64"
)

set_target_properties(LLVMhip64 PROPERTIES
                      RUNTIME_OUTPUT_DIRECTORY "${LLVM_LIBRARY_OUTPUT_INTDIR}/${OFFLOAD_TARGET_SUBDIR}"
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS LLVMhip64 ARCHIVE DESTINATION "${OFFLOAD_INSTALL_LIBDIR}"
        COMPONENT offload)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/hip/hip_runtime.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload/hip/)
