include(CheckLinkerFlag OPTIONAL)

if (COMMAND check_linker_flag)
  macro(llvm_check_linker_flag)
    check_linker_flag(${ARGN})
  endmacro()
else()
  # Until the minimum CMAKE version is 3.18

  include(CheckCXXCompilerFlag)
  include(CMakePushCheckState)

  # cmake builtin compatible, except we assume lang is C or CXX
  function(llvm_check_linker_flag lang flag out_var)
    cmake_push_check_state()
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
    if("${lang}" STREQUAL "C")
      check_c_compiler_flag("" ${out_var})
    elseif("${lang}" STREQUAL "CXX")
      check_cxx_compiler_flag("" ${out_var})
    else()
      message(FATAL_ERROR "\"${lang}\" is not C or CXX")
    endif()
    cmake_pop_check_state()
  endfunction()
endif()
