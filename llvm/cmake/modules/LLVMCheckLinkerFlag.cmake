include(CheckLinkerFlag OPTIONAL)

if (COMMAND check_linker_flag)
  macro(llvm_check_linker_flag)
    check_linker_flag(${ARGN})
  endmacro()
else()
  include(CheckCXXCompilerFlag)
  include(CMakePushCheckState)

  # cmake builtin compatible, except we assume lang is CXX
  function(llvm_check_linker_flag lang flag out_var)
    cmake_push_check_state()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
    check_cxx_compiler_flag("" ${out_var})
    cmake_pop_check_state()
  endfunction()
endif()
