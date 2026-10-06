cmake_minimum_required(VERSION 3.20)
project(LocalOptimization CXX)

# Configure with -DLLVM_DIR=<llvm-build>/lib/cmake/llvm
find_package(LLVM REQUIRED CONFIG)
set(CMAKE_CXX_STANDARD 17)

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFINITIONS_LIST})
include_directories(${LLVM_INCLUDE_DIRS})

# A plugin that opt loads at run time: build/LocalOptimization.so
add_library(LocalOptimization MODULE src/LocalOptimization.cpp)
set_target_properties(LocalOptimization PROPERTIES PREFIX "")
if(NOT LLVM_ENABLE_RTTI)
  target_compile_options(LocalOptimization PRIVATE -fno-rtti) # must match LLVM
endif()
