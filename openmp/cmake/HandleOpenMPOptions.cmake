if (NOT COMMAND append_if)
  # From HandleLLVMOptions.cmake
  function(append_if condition value)
    if (${condition})
      foreach(variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
      endforeach(variable)
    endif()
  endfunction()
endif()

if (NOT COMMAND append)
  function(append value)
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endfunction()
endif()

# MSVC and clang-cl in compatibility mode map -Wall to -Weverything.
# TODO: LLVM adds /W4 instead, check if that works for the OpenMP runtimes.
if (NOT MSVC)
  append_if(OPENMP_HAVE_WALL_FLAG "-Wall" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()
if (OPENMP_ENABLE_WERROR)
  append_if(OPENMP_HAVE_WERROR_FLAG "-Werror" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()

append_if(OPENMP_HAVE_COLOR_DIAGNOSTICS "-fcolor-diagnostics" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

# Additional warnings that are not enabled by -Wall.
append_if(OPENMP_HAVE_WCAST_QUAL_FLAG "-Wcast-qual" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
append_if(OPENMP_HAVE_WFORMAT_PEDANTIC_FLAG "-Wformat-pedantic" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
append_if(OPENMP_HAVE_WIMPLICIT_FALLTHROUGH_FLAG "-Wimplicit-fallthrough" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
append_if(OPENMP_HAVE_WSIGN_COMPARE_FLAG "-Wsign-compare" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

# Warnings that we want to disable because they are too verbose or fragile.

# GCC silently accepts any -Wno-<foo> option, but warns about those options
# being unrecognized only if the compilation triggers other warnings to be
# printed. Therefore, check for whether the compiler supports options in the
# form -W<foo>, and if supported, add the corresponding -Wno-<foo> option.

append_if(OPENMP_HAVE_WEXTRA_FLAG "-Wno-extra" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
append_if(OPENMP_HAVE_WPEDANTIC_FLAG "-Wno-pedantic" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
append_if(OPENMP_HAVE_WMAYBE_UNINITIALIZED_FLAG "-Wno-maybe-uninitialized" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

if (NOT (WIN32 OR CYGWIN))
  # This flag is not relevant on Windows; the flag is accepted, but produces warnings
  # about argument unused during compilation.
  append_if(OPENMP_HAVE_NO_SEMANTIC_INTERPOSITION "-fno-semantic-interposition" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()
append_if(OPENMP_HAVE_FUNCTION_SECTIONS "-ffunction-section" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
append_if(OPENMP_HAVE_DATA_SECTIONS "-fdata-sections" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

if (MSVC)
  # Disable "warning C4201: nonstandard extension used: nameless struct/union"
  append("-wd4201" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  
  # Disable "warning C4190: '__kmpc_atomic_cmplx8_rd' has C-linkage specified, but returns
  # UDT '__kmp_cmplx64_t' which is incompatible with C"
  append("-wd4190" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
endif()
