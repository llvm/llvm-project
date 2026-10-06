cmake_minimum_required(VERSION 3.20)
project(LocalOptimization LANGUAGES CXX)

# Configure with -DLLVM_DIR=<llvm-build-dir>/lib/cmake/llvm
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION} from ${LLVM_DIR}")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The plugin is loaded into opt at run time, so it is a MODULE library that
# does not link LLVM itself: opt already contains everything it uses.
add_library(LocalOptimization MODULE src/LocalOptimization.cpp)
set_target_properties(LocalOptimization PROPERTIES PREFIX "") # LocalOptimization.so

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
target_include_directories(LocalOptimization SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(LocalOptimization PRIVATE ${LLVM_DEFINITIONS_LIST})
target_compile_options(LocalOptimization PRIVATE -Wall -Wextra -Wno-unused-parameter)
# Must match how LLVM was built, or loading the plugin fails.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(LocalOptimization PRIVATE -fno-rtti)
endif()

# cmake --build build --target check
add_custom_target(check
  COMMAND ${CMAKE_COMMAND} -E env
          LLVM_BIN=${LLVM_TOOLS_BINARY_DIR}
          PLUGIN=$<TARGET_FILE:LocalOptimization>
          OUT_DIR=${CMAKE_BINARY_DIR}/testcases
          ${CMAKE_SOURCE_DIR}/testcases/run_tests.sh
  DEPENDS LocalOptimization
  USES_TERMINAL)
