message(STATUS "Building OpenACC offloading runtime library libacctarget.")

set(ACCTARGET_SRC
  Interface.cpp

  DeviceManager.cpp
  QueueManager.cpp

  RuntimeImpl.cpp
  RuntimeInterface.cpp
  DataRuntimeInterface.cpp
  CfiDataRuntimeInterface.cpp

)

add_llvm_library(acctarget
  SHARED

  ${ACCTARGET_SRC}

  ADDITIONAL_HEADER_DIRS
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}

  LINK_COMPONENTS
  Support

  LINK_LIBS
  PUBLIC
  ompaccsupport

  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)

target_include_directories(acctarget PRIVATE
  ${LIBOMPTARGET_INCLUDE_DIR}
  ${LIBOMPTARGET_BINARY_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(LLVM_HAVE_LINK_VERSION_SCRIPT)
  target_link_libraries(acctarget PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports")
  set_property(TARGET acctarget APPEND PROPERTY LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports)
endif()

add_dependencies(acctarget PluginErrcodes)

target_compile_definitions(acctarget PRIVATE
  TARGET_NAME=acctarget
  DEBUG_PREFIX="acctarget"
)

target_link_libraries(acctarget PRIVATE ompaccsupport)

target_compile_options(acctarget PRIVATE ${offload_compile_flags})
target_link_options(acctarget PRIVATE ${offload_link_flags})

# Link against flang_rt.runtime for Fortran descriptor support.
# flang_rt.runtime is a sibling runtime; link against the shared library target.
if(TARGET flang_rt.runtime.dynamic)
  target_link_libraries(acctarget PRIVATE flang_rt.runtime.dynamic)
elseif(TARGET flang_rt.runtime.static)
  target_link_libraries(acctarget PRIVATE flang_rt.runtime.static)
else()
  message(FATAL_ERROR "flang_rt.runtime target not found")
endif()

# libomptarget.so needs to be aware of where the plugins live as they
# are now separated in the build directory.
set_target_properties(acctarget PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      INSTALL_RPATH "$ORIGIN"
                      BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/..")
install(TARGETS acctarget LIBRARY COMPONENT acctarget DESTINATION "${OFFLOAD_INSTALL_LIBDIR}")
