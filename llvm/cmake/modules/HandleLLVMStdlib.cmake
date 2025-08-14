# This CMake module is responsible for setting the standard library to libc++
# if the user has requested it.

include(DetermineGCCCompatible)
include(CheckIncludeFiles)

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
      check_linker_flag(CXX "-stdlib=libc++" CXX_LINKER_SUPPORTS_STDLIB)

      # Check whether C++ include files are available
      # runtimes/CMakeLists.txt adds -nostdlib++ and -nostdinc++ to
      # CMAKE_REQUIRED_FLAGS, which are incompatible with -stdlib=libc++; use
      # a fresh CMAKE_REQUIRED_FLAGS environment.
      cmake_push_check_state(RESET)
      set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -stdlib=libc++")
      check_include_files("chrono" CXX_COMPILER_SUPPORTS_STDLIB_CHRONO LANGUAGE CXX)
      cmake_pop_check_state()

      if(CXX_COMPILER_SUPPORTS_STDLIB AND CXX_LINKER_SUPPORTS_STDLIB AND CXX_COMPILER_SUPPORTS_STDLIB_CHRONO)
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

  if(LLVM_STATIC_LINK_CXX_STDLIB)
    if(LLVM_COMPILER_IS_GCC_COMPATIBLE)
      check_cxx_compiler_flag("-static-libstdc++"
                              CXX_COMPILER_SUPPORTS_STATIC_STDLIB)
      check_linker_flag(CXX "-static-libstdc++" CXX_LINKER_SUPPORTS_STATIC_STDLIB)
      if(CXX_COMPILER_SUPPORTS_STATIC_STDLIB AND
        CXX_LINKER_SUPPORTS_STATIC_STDLIB)
        append("-static-libstdc++"
          CMAKE_EXE_LINKER_FLAGS CMAKE_SHARED_LINKER_FLAGS
          CMAKE_MODULE_LINKER_FLAGS)
      else()
        message(WARNING
          "Can't specify static linking for the C++ standard library")
      endif()
    else()
      message(WARNING "Not sure how to specify static linking of C++ standard "
        "library for this compiler")
    endif()
  endif()
endif()
