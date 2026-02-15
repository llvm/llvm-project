#
# Configure CMake flags related to C++ language.
#
# This file defines an interface target named `runtimes-language-flags` which
# can be linked against to get the language-related flags configured here.
#

add_library(runtimes-language-flags INTERFACE)

# Exceptions
option(RUNTIMES_ENABLE_EXCEPTIONS
  "Whether to enable exceptions (-fexceptions) when building the runtimes.
   This also controls whether the runtimes provide support for exceptions,
   for example stack unwinding and other exceptions-related features."
  ON)
if (RUNTIMES_ENABLE_EXCEPTIONS)
  if (MSVC)
    # Catches C++ exceptions only and tells the compiler to assume that extern C
    # functions never throw a C++ exception.
    target_compile_options(runtimes-language-flags INTERFACE "-EHsc")
  endif()
else()
  if (MSVC)
    target_compile_options(runtimes-language-flags INTERFACE "-EHs-" "-EHa-")
  else()
    target_compile_options(runtimes-language-flags INTERFACE "-fno-exceptions")
  endif()
endif()

# RTTI
option(RUNTIMES_ENABLE_RTTI
  "Whether to enable runtime type information (-frtti) when building the runtimes.
   Note that building the runtimes with support for exceptions requires RTTI to
   be enabled."
  ON)
if (NOT RUNTIMES_ENABLE_RTTI AND RUNTIMES_ENABLE_EXCEPTIONS)
  message(FATAL_ERROR "The runtimes cannot be built with exceptions enabled but RTTI disabled, since "
                      "that configuration is broken. See https://llvm.org/PR66117 for details.")
endif()
if (NOT RUNTIMES_ENABLE_RTTI)
  if (MSVC)
    target_compile_options(runtimes-language-flags INTERFACE -GR-)
  else()
    target_compile_options(runtimes-language-flags INTERFACE "-fno-rtti")
  endif()
endif()
