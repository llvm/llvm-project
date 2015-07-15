#
#//===----------------------------------------------------------------------===//
#//
#//                     The LLVM Compiler Infrastructure
#//
#// This file is dual licensed under the MIT and the University of Illinois Open
#// Source Licenses. See LICENSE.txt for details.
#//
#//===----------------------------------------------------------------------===//
#

function(libomp_get_definitions_flags cppflags)
  set(cppflags_local)
  libomp_append(cppflags_local "-D USE_ITT_BUILD")
  # yes... you need 5 backslashes...
  libomp_append(cppflags_local "-D KMP_ARCH_STR=\"\\\\\"${LIBOMP_LEGAL_ARCH}\\\\\"\"")
  libomp_append(cppflags_local "-D BUILD_I8")
  libomp_append(cppflags_local "-D KMP_LIBRARY_FILE=\\\\\"${LIBOMP_LIB_FILE}\\\\\"")
  libomp_append(cppflags_local "-D KMP_VERSION_MAJOR=${LIBOMP_VERSION}")
  libomp_append(cppflags_local "-D KMP_NESTED_HOT_TEAMS")

  # customize to 128 bytes for ppc64
  if(${PPC64})
    libomp_append(cppflags_local "-D CACHE_LINE=128")
  else()
    libomp_append(cppflags_local "-D CACHE_LINE=64")
  endif()

  libomp_append(cppflags_local "-D KMP_ADJUST_BLOCKTIME=1")
  libomp_append(cppflags_local "-D BUILD_PARALLEL_ORDERED")
  libomp_append(cppflags_local "-D KMP_ASM_INTRINS")
  libomp_append(cppflags_local "-D USE_ITT_NOTIFY" IF_TRUE_1_0 LIBOMP_USE_ITT_NOTIFY)
  libomp_append(cppflags_local "-D INTEL_NO_ITTNOTIFY_API" IF_FALSE LIBOMP_USE_ITT_NOTIFY)
  libomp_append(cppflags_local "-D INTEL_ITTNOTIFY_PREFIX=__kmp_itt_")
  libomp_append(cppflags_local "-D KMP_USE_VERSION_SYMBOLS" IF_TRUE LIBOMP_USE_VERSION_SYMBOLS)

  if(WIN32)
    libomp_append(cppflags_local "-D _CRT_SECURE_NO_WARNINGS")
    libomp_append(cppflags_local "-D _CRT_SECURE_NO_DEPRECATE")
    libomp_append(cppflags_local "-D _WINDOWS")
    libomp_append(cppflags_local "-D _WINNT")
    libomp_append(cppflags_local "-D _WIN32_WINNT=0x0501")
    libomp_append(cppflags_local "-D KMP_WIN_CDECL")
    libomp_append(cppflags_local "-D _USRDLL")
    libomp_append(cppflags_local "-D _ITERATOR_DEBUG_LEVEL=0" IF_TRUE DEBUG_BUILD)
  else()
    libomp_append(cppflags_local "-D _GNU_SOURCE")
    libomp_append(cppflags_local "-D _REENTRANT")
    libomp_append(cppflags_local "-D BUILD_TV")
    libomp_append(cppflags_local "-D USE_CBLKDATA")
    libomp_append(cppflags_local "-D KMP_GOMP_COMPAT")
  endif()

  libomp_append(cppflags_local "-D USE_LOAD_BALANCE" IF_FALSE MIC)
  if(NOT WIN32 AND NOT APPLE)
    libomp_append(cppflags_local "-D KMP_TDATA_GTID")
  endif()
  libomp_append(cppflags_local "-D KMP_USE_ASSERT" IF_TRUE LIBOMP_ENABLE_ASSERTIONS)
  libomp_append(cppflags_local "-D KMP_DYNAMIC_LIB")
  libomp_append(cppflags_local "-D KMP_STUB" IF_TRUE STUBS_LIBRARY)

  if(${DEBUG_BUILD} OR ${RELWITHDEBINFO_BUILD})
    libomp_append(cppflags_local "-D KMP_DEBUG")
  endif()
  libomp_append(cppflags_local "-D _DEBUG" IF_TRUE DEBUG_BUILD)
  libomp_append(cppflags_local "-D BUILD_DEBUG" IF_TRUE DEBUG_BUILD)
  libomp_append(cppflags_local "-D KMP_STATS_ENABLED" IF_TRUE_1_0 LIBOMP_STATS)
  libomp_append(cppflags_local "-D USE_DEBUGGER" IF_TRUE_1_0 LIBOMP_USE_DEBUGGER)
  libomp_append(cppflags_local "-D OMPT_SUPPORT" IF_TRUE_1_0 LIBOMP_OMPT_SUPPORT)
  libomp_append(cppflags_local "-D OMPT_BLAME" IF_TRUE_1_0 LIBOMP_OMPT_BLAME)
  libomp_append(cppflags_local "-D OMPT_TRACE" IF_TRUE_1_0 LIBOMP_OMPT_TRACE)

  # OpenMP version flags
  set(libomp_have_omp_50 0)
  set(libomp_have_omp_41 0)
  set(libomp_have_omp_40 0)
  set(libomp_have_omp_30 0)
  if(${LIBOMP_OMP_VERSION} EQUAL 50 OR ${LIBOMP_OMP_VERSION} GREATER 50)
    set(libomp_have_omp_50 1)
  endif()
  if(${LIBOMP_OMP_VERSION} EQUAL 41 OR ${LIBOMP_OMP_VERSION} GREATER 41)
    set(libomp_have_omp_41 1)
  endif()
  if(${LIBOMP_OMP_VERSION} EQUAL 40 OR ${LIBOMP_OMP_VERSION} GREATER 40)
    set(libomp_have_omp_40 1)
  endif()
  if(${LIBOMP_OMP_VERSION} EQUAL 30 OR ${LIBOMP_OMP_VERSION} GREATER 30)
    set(libomp_have_omp_30 1)
  endif()
  libomp_append(cppflags_local "-D OMP_50_ENABLED=${libomp_have_omp_50}")
  libomp_append(cppflags_local "-D OMP_41_ENABLED=${libomp_have_omp_41}")
  libomp_append(cppflags_local "-D OMP_40_ENABLED=${libomp_have_omp_40}")
  libomp_append(cppflags_local "-D OMP_30_ENABLED=${libomp_have_omp_30}")
  libomp_append(cppflags_local "-D KMP_USE_ADAPTIVE_LOCKS" IF_TRUE_1_0 LIBOMP_USE_ADAPTIVE_LOCKS)
  libomp_append(cppflags_local "-D KMP_DEBUG_ADAPTIVE_LOCKS=0")
  libomp_append(cppflags_local "-D KMP_USE_INTERNODE_ALIGNMENT" IF_TRUE_1_0 LIBOMP_USE_INTERNODE_ALIGNMENT)
  # CMake doesn't include CPPFLAGS from environment, but we will.
  set(${cppflags} ${cppflags_local} ${LIBOMP_CPPFLAGS} $ENV{CPPFLAGS} PARENT_SCOPE)
endfunction()

