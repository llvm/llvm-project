#===============================================================================
# Define targets for linking against the selected C library
#
# After including this file, the following targets are defined:
# - runtimes-libc-headers: An interface target that allows getting access to the
#                          headers of the selected C library.
# - runtimes-libc-shared: A target representing the selected shared C library.
# - runtimes-libc-static: A target representing the selected static C library.
#===============================================================================

include_guard(GLOBAL)

if(TARGET runtimes-libc-headers)
  return()
endif()

set(RUNTIMES_SUPPORTED_C_LIBRARIES system llvm-libc picolibc newlib)
set(RUNTIMES_USE_LIBC "system" CACHE STRING "Specify C library to use. Supported values are ${RUNTIMES_SUPPORTED_C_LIBRARIES}.")
if (NOT "${RUNTIMES_USE_LIBC}" IN_LIST RUNTIMES_SUPPORTED_C_LIBRARIES)
  message(FATAL_ERROR "Unsupported C library: '${RUNTIMES_USE_LIBC}'. Supported values are ${RUNTIMES_SUPPORTED_C_LIBRARIES}.")
endif()

# Link against a system-provided libc
if (RUNTIMES_USE_LIBC STREQUAL "system")
  add_library(runtimes-libc-headers INTERFACE)

  add_library(runtimes-libc-static INTERFACE)
  add_library(runtimes-libc-shared INTERFACE)

# Link against the in-tree LLVM libc
elseif (RUNTIMES_USE_LIBC STREQUAL "llvm-libc")
  add_library(runtimes-libc-headers INTERFACE)
  # libc-headers is a target only when LLVM libc is built in the same tree.
  # A runtime built against an LLVM libc that is already installed takes the
  # headers from where the compiler looks for them, as it would for any other
  # C library. Naming the missing target would put -llibc-headers on the link
  # line, and -nostdlibinc would hide the headers it needs.
  if (TARGET libc-headers)
    target_link_libraries(runtimes-libc-headers INTERFACE libc-headers)
    check_cxx_compiler_flag(-nostdlibinc CXX_SUPPORTS_NOSTDLIBINC_FLAG)
    if(CXX_SUPPORTS_NOSTDLIBINC_FLAG)
      target_compile_options(runtimes-libc-headers INTERFACE "-nostdlibinc")
      if(LIBC_KERNEL_HEADERS)
        target_compile_options(runtimes-libc-headers INTERFACE "-idirafter${LIBC_KERNEL_HEADERS}")
      endif()
    endif()
  endif()

  add_library(runtimes-libc-static INTERFACE)
  if (TARGET libc)
    target_link_libraries(runtimes-libc-static INTERFACE libc)
  endif()
  if (TARGET libm)
    target_link_libraries(runtimes-libc-static INTERFACE libm)
  endif()
  if (CXX_SUPPORTS_NOLIBC_FLAG)
    target_link_options(runtimes-libc-static INTERFACE "-nolibc")
  endif()

  # LLVM libc only builds shared objects when LIBC_ENABLE_SHARED is set, so
  # this stays empty otherwise and the runtimes fall back to a static link.
  add_library(runtimes-libc-shared INTERFACE)
  if (TARGET libc-shared)
    target_link_libraries(runtimes-libc-shared INTERFACE libc-shared)
  endif()
  if (TARGET libm-shared)
    target_link_libraries(runtimes-libc-shared INTERFACE libm-shared)
  endif()
  if (CXX_SUPPORTS_NOLIBC_FLAG AND (TARGET libc-shared OR TARGET libm-shared))
    target_link_options(runtimes-libc-shared INTERFACE "-nolibc")
  endif()
endif()
