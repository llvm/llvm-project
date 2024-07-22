set(supported_targets x86_64 aarch64 ppc64 ppc64le s390x)
if(NOT ${CMAKE_SYSTEM_PROCESSOR} IN_LIST supported_targets)
  message(STATUS "Not building ${machine} NextGen offloading plugin")
  return()
endif()

set(machine ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le$")
  set(machine ppc64)
endif()

# Create the library and add the default arguments.
add_target_library(omptarget.rtl.host ${machine})

target_sources(omptarget.rtl.host PRIVATE src/rtl.cpp)

if(LIBOMPTARGET_DEP_LIBFFI_FOUND)
  message(STATUS "Building ${machine} plugin linked with libffi")
  if(FFI_STATIC_LIBRARIES)
    target_link_libraries(omptarget.rtl.host PRIVATE FFI::ffi_static)
  else()
    target_link_libraries(omptarget.rtl.host PRIVATE FFI::ffi)
  endif()
else()
  message(STATUS "Building ${machine} plugin for dlopened libffi")
  target_sources(omptarget.rtl.host PRIVATE dynamic_ffi/ffi.cpp)
  target_include_directories(omptarget.rtl.host PRIVATE dynamic_ffi)
endif()

target_include_directories(omptarget.rtl.host PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

if(LIBOMPTARGET_DEP_LIBFFI_FOUND)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS omptarget.rtl.host)
  set(LIBOMPTARGET_TESTED_PLUGINS
      "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  message(STATUS "Not generating ${tmachine_name} tests. LibFFI not found.")
endif()

# Define the target specific triples and ELF machine values.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le$")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS
       "powerpc64le-ibm-linux-gnu" "powerpc64le-ibm-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64$")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS
       "powerpc64-ibm-linux-gnu" "powerpc64-ibm-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64$")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS
       "x86_64-pc-linux-gnu" "x86_64-pc-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64$")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS
       "aarch64-unknown-linux-gnu" "aarch64-unknown-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x$")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS
       "s390x-ibm-linux-gnu" "s390x-ibm-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
endif()
