cmake_minimum_required(VERSION 3.13)
project(MLIROBF LANGUAGES CXX C)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


find_package(MLIR REQUIRED CONFIG) # What this does is that it tells the code that I want to use MLIR. Go find it and link to it.


message(STATUS "Found MLIR: ${MLIR_PACKAGE_PREFIX}")


include_directories(${PROJECT_SOURCE_DIR}/include) # THis helps it find include/Obfuscator/Passes.h


add_subdirectory(lib) # This tells the CMake to build the code in  mlir-obfuscator/lib/


mlir_print_global_pass_registry() # Optional: Thjis is just to print MLIR passes for debugging
