include(CheckCXXCompilerFlag)

function(check_linker_flag flag out_var)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}")
  check_cxx_compiler_flag("" ${out_var})
endfunction()
