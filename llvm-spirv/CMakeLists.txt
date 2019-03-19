cmake_minimum_required(VERSION 3.3)

set(LLVM_SPIRV_VERSION 0.2.1.0)

# check if we build inside llvm or not
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(BUILD_EXTERNAL YES)
  project(LLVM_SPIRV
    VERSION
      ${LLVM_SPIRV_VERSION}
    LANGUAGES
      CXX
  )

  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  if(LLVM_INCLUDE_TESTS)
    set(LLVM_TEST_COMPONENTS
      llvm-as
      llvm-dis
    )
  endif(LLVM_INCLUDE_TESTS)

  find_package(LLVM 9.0.0 REQUIRED
    COMPONENTS
      Analysis
      BitReader
      BitWriter
      Core
      Support
      TransformUtils
      ${LLVM_TEST_COMPONENTS}
  )
  set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${LLVM_CMAKE_DIR}
  )
  include(AddLLVM)

  message(STATUS "Found LLVM: ${LLVM_VERSION}")
else(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(BUILD_EXTERNAL NO)
endif(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)

if(BUILD_EXTERNAL)
  find_program(CCACHE_EXE_FOUND ccache)
  if(CCACHE_EXE_FOUND)
    message(STATUS "Found ccache: ${CCACHE_EXE_FOUND}")
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
  endif()
endif()

set(LLVM_SPIRV_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(lib/SPIRV)
add_subdirectory(tools/llvm-spirv)
if(LLVM_INCLUDE_TESTS)
  add_subdirectory(test)
endif(LLVM_INCLUDE_TESTS)

install(
  FILES
    ${LLVM_SPIRV_INCLUDE_DIRS}/LLVMSPIRVLib.h
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/include/LLVMSPIRVLib
)

configure_file(LLVMSPIRVLib.pc.in ${CMAKE_BINARY_DIR}/LLVMSPIRVLib.pc @ONLY)
install(
  FILES
    ${CMAKE_BINARY_DIR}/LLVMSPIRVLib.pc
  DESTINATION
    ${CMAKE_INSTALL_PREFIX}/lib${LLVM_LIBDIR_SUFFIX}/pkgconfig
)
