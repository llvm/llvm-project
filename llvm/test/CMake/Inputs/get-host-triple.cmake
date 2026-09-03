# Driver for testing llvm/cmake/modules/GetHostTriple.cmake. It simulates a
# host platform by setting the variables that CMake's platform detection
# normally provides, then reports the triple computed by get_host_triple().
#
# Usage:
#   cmake -DSRC_ROOT=<llvm source root>
#         -DHOST_SYSTEM_NAME=<simulated CMAKE_HOST_SYSTEM_NAME>
#         -DHOST_SYSTEM_PROCESSOR=<simulated CMAKE_HOST_SYSTEM_PROCESSOR>
#         -P get-host-triple.cmake

if( NOT SRC_ROOT )
  message( FATAL_ERROR "SRC_ROOT is not set" )
endif()

set( CMAKE_HOST_SYSTEM_NAME "${HOST_SYSTEM_NAME}" )
set( CMAKE_HOST_SYSTEM_PROCESSOR "${HOST_SYSTEM_PROCESSOR}" )

include( "${SRC_ROOT}/cmake/modules/GetHostTriple.cmake" )

function( report_host_triple )
  get_host_triple( host_triple )
  message( "host_triple=${host_triple}" )
endfunction()

report_host_triple()
