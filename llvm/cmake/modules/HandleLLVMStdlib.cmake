# This CMake module is responsible for setting the standard library to libc++
# if the user has requested it.

include(DetermineGCCCompatible)

if(NOT DEFINED LLVM_STDLIB_HANDLED)
  set(LLVM_STDLIB_HANDLED ON)

  function(append value)
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endfunction()

  include(CheckCXXCompilerFlag)
  include(CheckLinkerFlag)
  set(LLVM_LIBCXX_USED 0)
  if(LLVM_ENABLE_LIBCXX)
    if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
      check_cxx_compiler_flag("-stdlib=libc++" CXX_COMPILER_SUPPORTS_STDLIB)
      check_linker_flag("-stdlib=libc++" CXX_LINKER_SUPPORTS_STDLIB)
      if(CXX_COMPILER_SUPPORTS_STDLIB AND CXX_LINKER_SUPPORTS_STDLIB)
        append("-stdlib=libc++"
          CMAKE_CXX_FLAGS CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
          CMAKE_MODULE_LINKER_FLAGS)
        set(LLVM_LIBCXX_USED 1)
      else()
        message(WARNING "Can't specify libc++ with '-stdlib='")
      endif()
    else()
      message(WARNING "Not sure how to specify libc++ for this compiler")
    endif()
  endif()
endif()
