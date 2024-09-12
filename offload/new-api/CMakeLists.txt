
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/API/OffloadAPI.td)
list(APPEND LLVM_TABLEGEN_FLAGS -I ${CMAKE_CURRENT_SOURCE_DIR}/API)

tablegen(OFFLOAD offload_api.h -gen-api)
tablegen(OFFLOAD offload_funcs.inc -gen-func-names)
tablegen(OFFLOAD offload_impl_func_decls.inc -gen-impl-func-decls)
tablegen(OFFLOAD offload_entry_points.inc -gen-entry-points)
tablegen(OFFLOAD offload_print.hpp -gen-print-header)


add_public_tablegen_target(OffloadHeaderGen)

add_llvm_library(offload_new SHARED
                src/offload_lib.cpp
                src/offload_impl.cpp
                DEPENDS OffloadHeaderGen
                LINK_LIBS omptarget omptarget.rtl.cuda omptarget.rtl.amdgpu)

target_include_directories(offload_new PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include)

# foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
#     target_link_libraries(offload_new PRIVATE omptarget.rtl.${plugin})
# endforeach()

set_target_properties(offload_new PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS offload_new LIBRARY COMPONENT offload_new DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/offload_api.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload)
