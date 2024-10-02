add_subdirectory(API)

add_llvm_library(offload_new SHARED
                src/offload_lib.cpp
                src/offload_impl.cpp)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
    target_link_libraries(offload_new PRIVATE omptarget.rtl.${plugin})
endforeach()

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
    target_link_libraries(offload_new PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/include/offload_exports")
endif()

target_include_directories(offload_new PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include)

target_compile_options(offload_new PRIVATE ${offload_compile_flags})
target_link_options(offload_new PRIVATE ${offload_link_flags})

set_target_properties(offload_new PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS offload_new LIBRARY COMPONENT offload_new DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/offload_api.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload)
