#
#//===----------------------------------------------------------------------===//
#//
#// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#// See https://llvm.org/LICENSE.txt for license information.
#// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#//
#//===----------------------------------------------------------------------===//
#

function(libomp_get_definitions_flags cppflags)
  set(cppflags_local)

  if(WIN32)
    libomp_append(cppflags_local "-D _CRT_SECURE_NO_WARNINGS")
    libomp_append(cppflags_local "-D _CRT_SECURE_NO_DEPRECATE")
    libomp_append(cppflags_local "-D _WINDOWS")
    libomp_append(cppflags_local "-D _WINNT")
    if (MSVC)
      # Force a default target OS version with MSVC based toolchains.
      # (For MinGW based ones, use the toolchain's default target or what
      # the user set in CMake flags.)
      libomp_append(cppflags_local "-D _WIN32_WINNT=0x0501")
    endif()
    libomp_append(cppflags_local "-D _USRDLL")
    libomp_append(cppflags_local "-D _ITERATOR_DEBUG_LEVEL=0" IF_TRUE DEBUG_BUILD)
    libomp_append(cppflags_local "-D _DEBUG" IF_TRUE DEBUG_BUILD)
  else()
    libomp_append(cppflags_local "-D _GNU_SOURCE")
    libomp_append(cppflags_local "-D _REENTRANT")
    libomp_append(cppflags_local "-D LIBOMP_HAVE_PTHREAD_SETNAME_NP" LIBOMP_HAVE_PTHREAD_SETNAME_NP)
    libomp_append(cppflags_local "-D LIBOMP_HAVE_PTHREAD_SET_NAME_NP" LIBOMP_HAVE_PTHREAD_SET_NAME_NP)
  endif()

  # CMake doesn't include CPPFLAGS from environment, but we will.
  set(${cppflags} ${cppflags_local} ${LIBOMP_CPPFLAGS} $ENV{CPPFLAGS} PARENT_SCOPE)
endfunction()
