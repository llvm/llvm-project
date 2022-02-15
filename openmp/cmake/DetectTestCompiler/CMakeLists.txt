cmake_minimum_required(VERSION 3.13.4)
project(DetectTestCompiler C CXX)

include(CheckCCompilerFlag)
include(CheckCXXCompilerFlag)

function(write_compiler_information lang)
  set(information "${CMAKE_${lang}_COMPILER}")
  set(information "${information}\\;${CMAKE_${lang}_COMPILER_ID}")
  set(information "${information}\\;${CMAKE_${lang}_COMPILER_VERSION}")
  set(information "${information}\\;${${lang}_FLAGS}")
  set(information "${information}\\;${${lang}_HAS_TSAN_FLAG}")
  set(information "${information}\\;${${lang}_HAS_OMIT_FRAME_POINTER}")
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${lang}CompilerInformation.txt ${information})
endfunction(write_compiler_information)

find_package(OpenMP)
if (NOT OpenMP_Found)
  set(OpenMP_C_FLAGS "-fopenmp")
  set(OpenMP_CXX_FLAGS "-fopenmp")
endif()

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

set(C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")
set(CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_THREAD_LIBS_INIT}")

# TODO: Implement blockaddress in GlobalISel and remove this flag!
if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
  check_c_compiler_flag("-fno-experimental-isel" C_HAS_EXPERIMENTAL_ISEL_FLAG)
  check_cxx_compiler_flag("-fno-experimental-isel" CXX_HAS_EXPERIMENTAL_ISEL_FLAG)
  macro(add_experimental_isel_flag lang)
    if (${lang}_HAS_EXPERIMENTAL_ISEL_FLAG)
      set(${lang}_FLAGS "-fno-experimental-isel ${${lang}_FLAGS}")
    endif()
  endmacro(add_experimental_isel_flag)

  add_experimental_isel_flag(C)
  add_experimental_isel_flag(CXX)
endif()

check_c_compiler_flag("-fno-omit-frame-pointer" C_HAS_OMIT_FRAME_POINTER)
check_cxx_compiler_flag("-fno-omit-frame-pointer" CXX_HAS_OMIT_FRAME_POINTER)

SET(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
check_c_compiler_flag("" C_HAS_TSAN_FLAG)
check_cxx_compiler_flag("" CXX_HAS_TSAN_FLAG)

write_compiler_information(C)
write_compiler_information(CXX)
