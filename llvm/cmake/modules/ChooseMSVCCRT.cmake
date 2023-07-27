# The macro choose_msvc_crt() takes a list of possible
# C runtimes to choose from, in the form of compiler flags,
# to present to the user. (MTd for /MTd, etc)
#
# The macro is invoked at the end of the file.
#
# This mechanism is deprecated, but kept for transitioning users.
#
# This reads the LLVM_USE_CRT_<CONFIG> options and sets
# CMAKE_MSVC_RUNTIME_LIBRARY accordingly. The previous mechanism allowed
# setting different choices for different build configurations (for
# multi-config generators), but translating multiple differing choices to
# the corresponding CMAKE_MSVC_RUNTIME_LIBRARY generator expression isn't
# supported by this transitional helper.

macro(choose_msvc_crt MSVC_CRT)
  if(LLVM_USE_CRT)
    message(FATAL_ERROR
      "LLVM_USE_CRT is deprecated. Use the CMAKE_BUILD_TYPE-specific
variables (LLVM_USE_CRT_DEBUG, etc) instead.")
  endif()

  foreach(build_type ${CMAKE_CONFIGURATION_TYPES} ${CMAKE_BUILD_TYPE})
    string(TOUPPER "${build_type}" build)
    if (NOT "${LLVM_USE_CRT_${build}}" STREQUAL "")
      if (NOT ${LLVM_USE_CRT_${build}} IN_LIST ${MSVC_CRT})
        message(FATAL_ERROR
          "Invalid value for LLVM_USE_CRT_${build}: ${LLVM_USE_CRT_${build}}. Valid options are one of: ${${MSVC_CRT}}")
      endif()
      set(library "MultiThreaded")
      if ("${LLVM_USE_CRT_${build}}" MATCHES "d$")
        set(library "${library}Debug")
      endif()
      if ("${LLVM_USE_CRT_${build}}" MATCHES "^MD")
        set(library "${library}DLL")
      endif()
      if(${runtime_library_set})
        message(WARNING "Conflicting LLVM_USE_CRT_* options")
      else()
        message(WARNING "The LLVM_USE_CRT_* options are deprecated, use the CMake provided CMAKE_MSVC_RUNTIME_LIBRARY setting instead")
      endif()
      set(CMAKE_MSVC_RUNTIME_LIBRARY "${library}" CACHE STRING "" FORCE)
      message(STATUS "Using VC++ CRT: ${CMAKE_MSVC_RUNTIME_LIBRARY}")
      set(runtime_library_set 1)
    endif()
  endforeach(build_type)
endmacro(choose_msvc_crt MSVC_CRT)


# List of valid CRTs for MSVC
set(MSVC_CRT
  MD
  MDd
  MT
  MTd)

choose_msvc_crt(MSVC_CRT)

