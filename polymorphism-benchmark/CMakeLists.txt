cmake_minimum_required(VERSION 3.20)
project(PolymorphismBenchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(BENCHMARK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third-party/benchmark"
  CACHE PATH "Google Benchmark source directory")

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(BENCHMARK_INSTALL_DOCS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(HAVE_STD_REGEX ON CACHE BOOL "" FORCE)
add_subdirectory("${BENCHMARK_SOURCE_DIR}" third-party/benchmark EXCLUDE_FROM_ALL)

add_executable(PolymorphismBM main.cpp Common.cpp)
target_link_libraries(PolymorphismBM PRIVATE benchmark::benchmark)

# Each style is compiled twice: once with inlinable methods and once with
# methods that behave as if defined in another translation unit.
foreach(style Virtual Classof Variant MLIR)
  foreach(inline 1 0)
    set(lib PolymorphismBM_${style}_${inline})
    add_library(${lib} OBJECT ${style}.cpp)
    target_compile_definitions(${lib} PRIVATE POLY_INLINE=${inline})
    target_link_libraries(${lib} PRIVATE benchmark::benchmark)
    target_link_libraries(PolymorphismBM PRIVATE ${lib})
  endforeach()
endforeach()
