add_subdirectory(API)

add_library(LLVMOffload SHARED
  src/OffloadLib.cpp
  src/OffloadImpl.cpp
)
add_dependencies(LLVMOffload OffloadAPI PluginErrcodes)

if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs FrontendOpenMP Support)
endif()
target_link_libraries(LLVMOffload PRIVATE ${llvm_libs})

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
    target_link_libraries(LLVMOffload PRIVATE omptarget.rtl.${plugin})
endforeach()

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
    target_link_libraries(LLVMOffload PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_include_directories(LLVMOffload PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}/API
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include
                            ${CMAKE_CURRENT_BINARY_DIR}/../plugins-nextgen/common/include
                            )

target_compile_options(LLVMOffload PRIVATE ${offload_compile_flags})
target_link_options(LLVMOffload PRIVATE ${offload_link_flags})

target_compile_definitions(LLVMOffload PRIVATE
  TARGET_NAME="Liboffload"
  DEBUG_PREFIX="Liboffload"
)

# Don't override an externally defined RPATH
if(NOT DEFINED CMAKE_INSTALL_RPATH)
  set_target_properties(LLVMOffload PROPERTIES
                        POSITION_INDEPENDENT_CODE ON
                        INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib"
                        BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
else()
  set_target_properties(LLVMOffload PROPERTIES
                        POSITION_INDEPENDENT_CODE ON
                        INSTALL_RPATH ${CMAKE_INSTALL_RPATH}
                        BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
endif()
install(TARGETS LLVMOffload LIBRARY COMPONENT offload DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/API/OffloadAPI.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload COMPONENT offload)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/API/OffloadPrint.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload COMPONENT offload)
