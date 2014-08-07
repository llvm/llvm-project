set(OMP_SHLIBEXT "${CMAKE_SHARED_LIBRARY_SUFFIX}")

execute_process(COMMAND "date" "+%Y-%m-%d %H:%M:%S" OUTPUT_VARIABLE BUILD_TIME)
string(REGEX REPLACE "\n" "" BUILD_TIME ${BUILD_TIME})

include_directories(
  .
  include
  include/${OMP_VERSION}
  i18n
  thirdparty/ittnotify
  ${CMAKE_CURRENT_BINARY_DIR}
)

if(WIN32)
  set(OS_GEN "win")
elseif(APPLE)
  set(OS_GEN "mac")
elseif(UNIX)
  set(OS_GEN "lin")
else()
  message(FATAL_ERROR "Unsupported OS")
endif()

if(arch)	
	set(ARCH ${arch}) #acquire from command line
else() #assume default
  set(ARCH "32e")
endif()

if("${arch}" STREQUAL "32e")
  set(ARCH_STR "Intel(R) 64")
elseif("${arch}" STREQUAL "ppc64")
  set(ARCH_STR "PPC64")
endif()

set(FEATURE_FLAGS "-D USE_ITT_BUILD")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D NDEBUG")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_ARCH_STR=\"\\\"${ARCH_STR}\\\"\"")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D _GNU_SOURCE")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D _REENTRANT")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_USE_ASSERT")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D BUILD_I8")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D BUILD_TV")

if(APPLE)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -current_version 5.0")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -compatibility_version 5.0")
endif()

set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_LIBRARY_FILE=\\\"libiomp5.${OMP_SHLIBEXT}\\\"")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_VERSION_MAJOR=${VERSION}")

# customize cache line size for ppc64 to 128 bytes: 64 in all other cases
if("${arch}" STREQUAL "ppc64")
	set(FEATURE_FLAGS "${FEATURE_FLAGS} -D CACHE_LINE=128")
else()
	set(FEATURE_FLAGS "${FEATURE_FLAGS} -D CACHE_LINE=64")	
endif()

set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_ADJUST_BLOCKTIME=1")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D BUILD_PARALLEL_ORDERED")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_ASM_INTRINS")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D USE_LOAD_BALANCE")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D USE_CBLKDATA")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D GUIDEDLL_EXPORTS")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_GOMP_COMPAT")

#adaptive locks use x86 assembly - disable for ppc64
if("${arch}" STREQUAL "32e" OR "${arch}" STREQUAL "32" OR "${arch}" STREQUAL "arm")
	set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_USE_ADAPTIVE_LOCKS=1")
	set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_DEBUG_ADAPTIVE_LOCKS=0")
endif()

set(FEATURE_FLAGS "${FEATURE_FLAGS} -D OMP_50_ENABLED=0")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D OMP_41_ENABLED=0")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D OMP_40_ENABLED=1")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D OMP_30_ENABLED=1")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D USE_ITT_NOTIFY=1")
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D INTEL_ITTNOTIFY_PREFIX=__kmp_itt_")
if (NOT APPLE)
  set(FEATURE_FLAGS "${FEATURE_FLAGS} -D KMP_TDATA_GTID")
endif()
set(FEATURE_FLAGS "${FEATURE_FLAGS} -D _KMP_BUILD_TIME=\"\\\"${BUILD_TIME} UTC\\\"\"")

set(COMMON_FLAGS "-fPIC")
set(COMMON_FLAGS "${COMMON_FLAGS} -Wno-unused-value")
set(COMMON_FLAGS "${COMMON_FLAGS} -Wno-switch")
set(COMMON_FLAGS "${COMMON_FLAGS} -Wno-deprecated-register")
set(COMMON_FLAGS "${COMMON_FLAGS} -fno-exceptions")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FEATURE_FLAGS} ${COMMON_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${FEATURE_FLAGS} ${COMMON_FLAGS}")

set(ASM_SOURCES
  z_Linux_asm.s
)
set(SOURCES
  thirdparty/ittnotify/ittnotify_static.c
  kmp_affinity.cpp
  kmp_alloc.c
  kmp_atomic.c
  kmp_cancel.cpp
  kmp_csupport.c
  kmp_debug.c
  kmp_dispatch.cpp
  kmp_environment.c
  kmp_error.c
  kmp_ftn_cdecl.c
  kmp_ftn_extra.c
  kmp_global.c
  kmp_gsupport.c
  kmp_i18n.c
  kmp_io.c
  kmp_itt.c
  kmp_lock.cpp
  kmp_runtime.c
  kmp_sched.cpp
  kmp_settings.c
  kmp_str.c
  kmp_taskdeps.cpp
  kmp_tasking.c
  kmp_taskq.c
  kmp_threadprivate.c
  kmp_utility.c
  kmp_version.c
  z_Linux_util.c
)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${ASM_SOURCES} PROPERTIES LANGUAGE CXX)

add_custom_command(
  OUTPUT kmp_i18n_id.inc
    COMMAND perl ${CMAKE_CURRENT_SOURCE_DIR}/../tools/message-converter.pl --os=${OS_GEN} --arch=${ARCH} --prefix=kmp_i18n --enum=kmp_i18n_id.inc ${CMAKE_CURRENT_SOURCE_DIR}/i18n/en_US.txt
)
add_custom_command(
  OUTPUT kmp_i18n_default.inc
    COMMAND perl ${CMAKE_CURRENT_SOURCE_DIR}/../tools/message-converter.pl --os=${OS_GEN} --arch=${ARCH} --prefix=kmp_i18n --default=kmp_i18n_default.inc ${CMAKE_CURRENT_SOURCE_DIR}/i18n/en_US.txt
)
add_custom_command(
  OUTPUT omp.h
    COMMAND perl ${CMAKE_CURRENT_SOURCE_DIR}/../tools/expand-vars.pl --strict -D Revision=\"\\$$Revision\" -D Date=\"\\$$Date\" -D KMP_TYPE=\"Performance\" -D KMP_ARCH=\"\\\"${ARCH_STR}\\\"\" -D KMP_VERSION_MAJOR=${VERSION} -D KMP_VERSION_MINOR=0 -D KMP_VERSION_BUILD=00000000 -D KMP_BUILD_DATE=\"${BUILD_TIME} UTC\" -D KMP_TARGET_COMPILER=12 -D KMP_DIAG=0 -D KMP_DEBUG_INFO=0 -D OMP_VERSION=${OMP_VERSION} ${CMAKE_CURRENT_SOURCE_DIR}/include/${OMP_VERSION_NUM}/omp.h.var omp.h
)

if("${ARCH}" STREQUAL "ppc64")
	add_custom_command(
		OUTPUT z_Linux_asm.o
    		       COMMAND ${CMAKE_CXX_COMPILER} -c -o z_Linux_asm.o -D KMP_ASM_INTRINS -D KMP_GOMP_COMPAT -D KMP_ARCH_PPC64 -x assembler-with-cpp ${CMAKE_CURRENT_SOURCE_DIR}/${ASM_SOURCES}
		       )
else()
	add_custom_command(
		OUTPUT z_Linux_asm.o
    		       COMMAND ${CMAKE_CXX_COMPILER} -c -o z_Linux_asm.o -D KMP_ASM_INTRINS -D KMP_GOMP_COMPAT -D KMP_ARCH_X86_64 -x assembler-with-cpp ${CMAKE_CURRENT_SOURCE_DIR}/${ASM_SOURCES}
		       )
endif()


add_custom_target(gen_kmp_i18n DEPENDS kmp_i18n_id.inc kmp_i18n_default.inc omp.h z_Linux_asm.o)

if(NOT APPLE)
	if(${ARCH} STREQUAL "ppc64" AND ${OS_GEN} STREQUAL "lin")
  		set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports_so.txt -ldl")
 	else()
   		set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports_so.txt")
	endif()
endif()

add_library(iomp5 SHARED ${SOURCES} z_Linux_asm.o)

# This is a workaround to a known ppc64 issue about libpthread. For more
# information see
# http://ryanarn.blogspot.com/2011/07/curious-case-of-pthreadatfork-on.html
if("${ARCH}" STREQUAL "ppc64")
	      find_library(PTHREAD NAMES pthread)
	      target_link_libraries(iomp5 ${PTHREAD})
endif()

add_dependencies(iomp5 gen_kmp_i18n)
