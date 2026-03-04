# - Adds a compiler flag if it is supported by the compiler
#
# This function checks that the supplied compiler flag is supported and then
# adds it to the corresponding compiler flags
#
#  add_cxx_compiler_flag(<FLAG> [<VARIANT>])
#
# - Example
#
# include(AddCXXCompilerFlag)
# add_cxx_compiler_flag(-Wall)
# add_cxx_compiler_flag(-no-strict-aliasing RELEASE)
# Requires CMake 2.6+

if(__add_cxx_compiler_flag)
  return()
endif()
set(__add_cxx_compiler_flag INCLUDED)

include(CheckCXXCompilerFlag)

function(mangle_compiler_flag FLAG OUTPUT)
  string(TOUPPER "HAVE_CXX_FLAG_${FLAG}" SANITIZED_FLAG)
  string(REPLACE "+" "X" SANITIZED_FLAG ${SANITIZED_FLAG})
  string(REGEX REPLACE "[^A-Za-z_0-9]" "_" SANITIZED_FLAG ${SANITIZED_FLAG})
  string(REGEX REPLACE "_+" "_" SANITIZED_FLAG ${SANITIZED_FLAG})
  set(${OUTPUT} "${SANITIZED_FLAG}" PARENT_SCOPE)
endfunction(mangle_compiler_flag)

set(BENCHMARK_GCC_COMPATIBLE_FLAGS
  -fno-exceptions
  -fstrict-aliasing
  -pedantic
  -pedantic-errors
  -Wall
  -Werror
  -Wextra
  -Wfloat-equal
  -Wno-deprecated
  -Wno-deprecated-declarations
  -Wold-style-cast
  -Wshadow
  -Wstrict-aliasing
  -Wsuggest-override
  )

macro(_benchmark_populate_cxx_compiler_flag)
  if(ARGC GREATER 1)
    set(VARIANT ${ARGV1})
    string(TOUPPER "_${VARIANT}" VARIANT)
  else()
    set(VARIANT "")
  endif()
  set(CMAKE_CXX_FLAGS${VARIANT} "${CMAKE_CXX_FLAGS${VARIANT}} ${BENCHMARK_CXX_FLAGS${VARIANT}} ${FLAG}" PARENT_SCOPE)
endmacro()

function(add_cxx_compiler_flag FLAG)
  mangle_compiler_flag("${FLAG}" MANGLED_FLAG)
  if(LLVM_COMPILER_IS_GCC_COMPATIBLE AND "${FLAG}" IN_LIST BENCHMARK_GCC_COMPATIBLE_FLAGS)
    _benchmark_populate_cxx_compiler_flag(${FLAG} ${VARIANT})
    return()
  endif()
  set(OLD_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${FLAG}")
  check_cxx_compiler_flag("${FLAG}" ${MANGLED_FLAG})
  set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS}")
  if(${MANGLED_FLAG})
    _benchmark_populate_cxx_compiler_flag(${FLAG} ${VARIANT})
  endif()
endfunction()

function(add_required_cxx_compiler_flag FLAG)
  mangle_compiler_flag("${FLAG}" MANGLED_FLAG)
  set(OLD_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${FLAG}")
  check_cxx_compiler_flag("${FLAG}" ${MANGLED_FLAG})
  set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS}")
  if(${MANGLED_FLAG})
    if(ARGC GREATER 1)
      set(VARIANT ${ARGV1})
      string(TOUPPER "_${VARIANT}" VARIANT)
    else()
      set(VARIANT "")
    endif()
    set(CMAKE_CXX_FLAGS${VARIANT} "${CMAKE_CXX_FLAGS${VARIANT}} ${FLAG}" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${FLAG}" PARENT_SCOPE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${FLAG}" PARENT_SCOPE)
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${FLAG}" PARENT_SCOPE)
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} ${FLAG}" PARENT_SCOPE)
  else()
    message(FATAL_ERROR "Required flag '${FLAG}' is not supported by the compiler")
  endif()
endfunction()

function(check_cxx_warning_flag FLAG)
  mangle_compiler_flag("${FLAG}" MANGLED_FLAG)
  set(OLD_CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS}")
  # Add -Werror to ensure the compiler generates an error if the warning flag
  # doesn't exist.
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror ${FLAG}")
  check_cxx_compiler_flag("${FLAG}" ${MANGLED_FLAG})
  set(CMAKE_REQUIRED_FLAGS "${OLD_CMAKE_REQUIRED_FLAGS}")
endfunction()
