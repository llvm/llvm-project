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
    libomp_append(cppflags_local "-D _WIN32_WINNT=0x0501")
    libomp_append(cppflags_local "-D _USRDLL")
    libomp_append(cppflags_local "-D _ITERATOR_DEBUG_LEVEL=0" IF_TRUE DEBUG_BUILD)
    libomp_append(cppflags_local "-D _DEBUG" IF_TRUE DEBUG_BUILD)
  else()
    libomp_append(cppflags_local "-D _GNU_SOURCE")
    libomp_append(cppflags_local "-D _REENTRANT")
  endif()

  # CMake doesn't include CPPFLAGS from environment, but we will.
  set(${cppflags} ${cppflags_local} ${LIBOMP_CPPFLAGS} $ENV{CPPFLAGS} PARENT_SCOPE)
endfunction()

