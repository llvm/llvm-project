message(STATUS "Building OpenACC offloading runtime library libacctarget.")

remove_definitions(-DOMPT_SUPPORT=1)

add_library(acctarget SHARED
  Interface.cpp
  RuntimeState.cpp

  ${LIBOMPACCSUPPORT_SOURCES}
)

if(LLVM_LINK_LLVM_DYLIB)
  set(llvm_libs LLVM)
else()
  llvm_map_components_to_libnames(llvm_libs FrontendOpenMP Support)
endif()
target_link_libraries(acctarget PRIVATE ${llvm_libs})
target_include_directories(acctarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR} ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../plugins-nextgen/common/include
  ${CMAKE_CURRENT_BINARY_DIR}/../plugins-nextgen/common/include
)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(acctarget PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
endif()

# Define the TARGET_NAME and DEBUG_PREFIX.
target_compile_definitions(acctarget PRIVATE
  TARGET_NAME=acctarget
  DEBUG_PREFIX="acctarget"
)


target_link_libraries(acctarget PRIVATE LLVMOffloadStatic)

target_compile_options(acctarget PRIVATE ${offload_compile_flags})
target_link_options(acctarget PRIVATE ${offload_link_flags})

# libacctarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(acctarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "${OFFLOAD_INSTALL_RPATH}"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
if(UNIX AND NOT APPLE)
  set_target_properties(acctarget PROPERTIES
                        SOVERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}"
                        VERSION "${LLVM_VERSION_MAJOR}.${LLVM_VERSION_MINOR}${LLVM_VERSION_SUFFIX}")
endif()
install(TARGETS acctarget LIBRARY COMPONENT offload DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
