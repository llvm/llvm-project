if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
  return()
endif()

  # build_generic_elf64("s390x" "S390X" "s390x" "systemz" "s390x-ibm-linux-gnu" "22")
  # build_generic_elf64("aarch64" "aarch64" "aarch64" "aarch64" "aarch64-unknown-linux-gnu" "183")
  # build_generic_elf64("ppc64" "PPC64" "ppc64" "ppc64" "powerpc64-ibm-linux-gnu" "21")
  # build_generic_elf64("x86_64" "x86_64" "x86_64" "x86_64" "x86_64-pc-linux-gnu" "62")
  # build_generic_elf64("ppc64le" "PPC64le" "ppc64" "ppc64le" "powerpc64le-ibm-linux-gnu" "21")
set(supported_targets x86_64 aarch64 ppc64 ppc64le s390x)
if(NOT ${CMAKE_SYSTEM_PROCESSOR} IN_LIST supported_targets)
  libomptarget_say("Not building ${machine} NextGen offloading plugin")
  return()
endif()

set(machine ${CMAKE_SYSTEM_PROCESSOR})
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le$")
  set(machine ppc64)
endif()

add_llvm_library(omptarget.rtl.${machine} SHARED
  src/rtl.cpp
  ADDITIONAL_HEADER_DIRS
    ${LIBOMPTARGET_INCLUDE_DIR}
  LINK_LIBS PRIVATE
    PluginCommon
    ${OPENMP_PTHREAD_LIB}
  NO_INSTALL_RPATH
  BUILDTREE_ONLY
)

if(LIBOMPTARGET_DEP_LIBFFI_FOUND)
  libomptarget_say("Building ${machine} plugin linked with libffi")
  if(FFI_STATIC_LIBRARIES)
    target_link_libraries(omptarget.rtl.${machine} PRIVATE FFI::ffi_static)
  else()
    target_link_libraries(omptarget.rtl.${machine} PRIVATE FFI::ffi)
  endif()
else()
  libomptarget_say("Building ${machine} plugin for dlopened libffi")
  target_sources(omptarget.rtl.${machine} PRIVATE dynamic_ffi/ffi.cpp)
  target_include_directories(omptarget.rtl.${machine} PRIVATE dynamic_ffi)
endif()

if(OMPT_TARGET_DEFAULT AND LIBOMPTARGET_OMPT_SUPPORT)
  target_link_libraries(omptarget.rtl.${machine} PRIVATE OMPT)
endif()

if(LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget.rtl.${machine} PRIVATE
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports")
endif()

# Install plugin under the lib destination folder.
install(TARGETS omptarget.rtl.${machine}
        LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.${machine} PROPERTIES
  INSTALL_RPATH "$ORIGIN" BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET protected)

target_include_directories(omptarget.rtl.${machine} PRIVATE
                           ${LIBOMPTARGET_INCLUDE_DIR})

if(LIBOMPTARGET_DEP_LIBFFI_FOUND)
  list(APPEND LIBOMPTARGET_TESTED_PLUGINS omptarget.rtl.${machine})
  set(LIBOMPTARGET_TESTED_PLUGINS
      "${LIBOMPTARGET_TESTED_PLUGINS}" PARENT_SCOPE)
else()
  libomptarget_say("Not generating ${tmachine_name} tests. LibFFI not found.")
endif()

# Define macro to be used as prefix of the runtime messages for this target.
target_compile_definitions(omptarget.rtl.${machine} PRIVATE TARGET_NAME=${machine})
# TODO: This should be automatized in Debug.h.
target_compile_definitions(omptarget.rtl.${machine} PRIVATE
                           DEBUG_PREFIX="TARGET ${machine} RTL")

# Define the target specific triples and ELF machine values.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le$" OR
   CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64$")
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE TARGET_ELF_ID=EM_PPC64)
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE
      LIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE="powerpc64-ibm-linux-gnu")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS 
       "powerpc64-ibm-linux-gnu" "powerpc64-ibm-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64$")
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE TARGET_ELF_ID=EM_X86_64)
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE
      LIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE="x86_64-pc-linux-gnu")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS 
       "x86_64-pc-linux-gnu" "x86_64-pc-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64$")
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE TARGET_ELF_ID=EM_AARCH64)
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE
      LIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE="aarch64-unknown-linux-gnu")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS 
       "aarch64-unknown-linux-gnu" "aarch64-unknown-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x$")
  target_compile_definitions(omptarget.rtl.${machine} TARGET_ELF_ID=EM_S390)
  target_compile_definitions(omptarget.rtl.${machine} PRIVATE
      LIBOMPTARGET_NEXTGEN_GENERIC_PLUGIN_TRIPLE="s390x-ibm-linux-gnu")
  list(APPEND LIBOMPTARGET_SYSTEM_TARGETS 
       "s390x-ibm-linux-gnu" "s390x-ibm-linux-gnu-LTO")
  set(LIBOMPTARGET_SYSTEM_TARGETS "${LIBOMPTARGET_SYSTEM_TARGETS}" PARENT_SCOPE)
endif()
