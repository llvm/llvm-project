add_subdirectory(API)

add_llvm_library(
  LLVMOffload SHARED
  src/OffloadLib.cpp
  src/OffloadImpl.cpp

  LINK_COMPONENTS
  FrontendOpenMP
  Support
  )

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
    target_link_libraries(LLVMOffload PRIVATE omptarget.rtl.${plugin})
endforeach()

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
    target_link_libraries(LLVMOffload PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_include_directories(LLVMOffload PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include/generated
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include)

target_compile_options(LLVMOffload PRIVATE ${offload_compile_flags})
target_link_options(LLVMOffload PRIVATE ${offload_link_flags})

target_compile_definitions(LLVMOffload PRIVATE
  TARGET_NAME="Liboffload"
  DEBUG_PREFIX="Liboffload"
)

set_target_properties(LLVMOffload PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS LLVMOffload LIBRARY COMPONENT LLVMOffload DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generated/OffloadAPI.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/generated/OffloadPrint.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload)
