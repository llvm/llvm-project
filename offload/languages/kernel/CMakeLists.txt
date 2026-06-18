add_llvm_library(
  LLVMOffloadKernel SHARED

  ../kernel/src/State.cpp
  ../kernel/src/ExportedAPI.cpp

  LINK_COMPONENTS
  Support
  Offload
  )

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(LLVMOffloadKernel PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_include_directories(LLVMOffloadKernel PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../liboffload/include/generated
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../../plugins-nextgen/common/include)

target_compile_options(LLVMOffloadKernel PRIVATE ${offload_compile_flags})
target_link_options(LLVMOffloadKernel PRIVATE ${offload_link_flags})

target_compile_definitions(LLVMOffloadKernel PRIVATE
  TARGET_NAME="libLLVMOffloadKernel"
  DEBUG_PREFIX="LLVMOffloadKernel"
)

set_target_properties(LLVMOffloadKernel PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS LLVMOffloadKernel LIBRARY COMPONENT LLVMOffloadKernel DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/DefineLanguageNames.inc DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload/kernel/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/LanguageRuntime.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload/kernel/)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../include/kernel/UndefineLanguageNames.inc DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload/kernel/)
