add_subdirectory(API)

add_library(LLVMOffloadObjects OBJECT
  src/OffloadLib.cpp
  src/OffloadImpl.cpp
)
add_dependencies(LLVMOffloadObjects OffloadAPI PluginErrcodes)
set_target_properties(LLVMOffloadObjects PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs FrontendOpenMP Support)
endif()

target_link_libraries(LLVMOffloadObjects PRIVATE PluginCommon)

foreach(plugin IN LISTS LIBOMPTARGET_PLUGINS_TO_BUILD)
    target_link_libraries(LLVMOffloadObjects PRIVATE omptarget.rtl.${plugin})
endforeach()


target_include_directories(LLVMOffloadObjects PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}/API
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include
                            PRIVATE
                            ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include
                            ${CMAKE_CURRENT_BINARY_DIR}/../plugins-nextgen/common/include
                            )
target_compile_options(LLVMOffloadObjects PRIVATE ${offload_compile_flags})
target_compile_definitions(LLVMOffloadObjects PRIVATE
  TARGET_NAME="Liboffload"
  DEBUG_PREFIX="Liboffload"
)

add_library(LLVMOffloadStatic STATIC)
target_link_libraries(LLVMOffloadStatic PUBLIC LLVMOffloadObjects ${llvm_libs})
set_target_properties(LLVMOffloadStatic PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(LLVMOffload SHARED)
target_link_libraries(LLVMOffload PRIVATE LLVMOffloadObjects ${llvm_libs})
target_include_directories(LLVMOffload PUBLIC
                            ${CMAKE_CURRENT_BINARY_DIR}/API
                            ${CMAKE_CURRENT_BINARY_DIR}/../include
                            ${CMAKE_CURRENT_SOURCE_DIR}/include
                            ${CMAKE_CURRENT_SOURCE_DIR}/../include
                            )

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(LLVMOffload PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

target_link_options(LLVMOffload PRIVATE ${offload_link_flags})

set_target_properties(LLVMOffload PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "${OFFLOAD_INSTALL_RPATH}"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
if(UNIX AND NOT APPLE)
  set_target_properties(LLVMOffload PROPERTIES
                        SOVERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}"
                        VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}")
endif()
install(TARGETS LLVMOffload
  COMPONENT offload
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  LIBRARY DESTINATION "${OFFLOAD_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")

if(MSVC)
  install(FILES "$<TARGET_PDB_FILE:LLVMOffload>"
    CONFIGURATIONS Debug RelWithDebInfo
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    OPTIONAL)
endif()

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/API/OffloadAPI.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload COMPONENT offload)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/API/OffloadPrint.hpp DESTINATION ${CMAKE_INSTALL_PREFIX}/include/offload COMPONENT offload)
