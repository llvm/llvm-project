set(SOURCES
  BasicBlockTracing.c
  CommonProfiling.c
  GCDAProfiling.c
  PathProfiling.c
  EdgeProfiling.c
  OptimalEdgeProfiling.c
  Profiling.h
  )

add_llvm_library( profile_rt-static ${SOURCES} )
set_target_properties( profile_rt-static
  PROPERTIES
  OUTPUT_NAME "profile_rt" )

set(BUILD_SHARED_LIBS ON)
add_llvm_library( profile_rt-shared ${SOURCES} )
set_target_properties( profile_rt-shared
  PROPERTIES
  OUTPUT_NAME "profile_rt" )
