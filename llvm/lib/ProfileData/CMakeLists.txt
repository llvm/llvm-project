add_llvm_component_library(LLVMProfileData
  DataAccessProf.cpp
  ETMTraceDecoder.cpp
  GCOV.cpp
  IndexedMemProfData.cpp
  InstrProf.cpp
  InstrProfCorrelator.cpp
  InstrProfReader.cpp
  InstrProfWriter.cpp
  ItaniumManglingCanonicalizer.cpp
  MemProf.cpp
  MemProfCommon.cpp
  MemProfReader.cpp
  MemProfRadixTree.cpp
  MemProfSummary.cpp
  MemProfSummaryBuilder.cpp
  PGOCtxProfReader.cpp
  PGOCtxProfWriter.cpp
  ProfileSummaryBuilder.cpp
  SampleProf.cpp
  SampleProfReader.cpp
  SampleProfWriter.cpp
  SymbolRemappingReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ProfileData

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  BitstreamReader
  Core
  Object
  Support
  Demangle
  Symbolize
  DebugInfoDWARF
  DebugInfoDWARFLowLevel
  TargetParser
  )

set(LLVM_ENABLE_OPENCSD "OFF" CACHE STRING "Use OpenCSD if available. Can be ON, OFF, or FORCE_ON")

set(LLVM_HAVE_OPENCSD OFF CACHE BOOL "Is OpenCSD available" FORCE)

# Minimum OpenCSD release that exposes every symbol referenced by
# ETMTraceDecoder.cpp. OCSD_OPFLG_CHK_RANGE_CONTINUE was introduced in
# OpenCSD 1.5.4; older releases (e.g. the 1.2.0 shipped by Debian/Ubuntu)
# build the OpenCSD library but cannot satisfy our use of the C API.
set(LLVM_OPENCSD_MIN_VERSION "1.5.4")

if(NOT LLVM_ENABLE_OPENCSD STREQUAL "OFF")
  find_library(OPENCSD_LIB NAMES opencsd_c_api opencsd)
  find_path(OPENCSD_INCLUDE NAMES opencsd/ocsd_if_types.h)

  set(OPENCSD_VERSION "")
  set(OPENCSD_VERSION_OK OFF)
  if(OPENCSD_LIB AND OPENCSD_INCLUDE)
    # The header exposes an OCSD_VER_STRING macro of the form "M.m.p"
    # which we extract for a robust version comparison.
    file(READ "${OPENCSD_INCLUDE}/opencsd/ocsd_if_version.h" _opencsd_ver_h)
    string(REGEX MATCH "OCSD_VER_STRING[ \t]+\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
                 _ver_match "${_opencsd_ver_h}")
    set(OPENCSD_VERSION "${CMAKE_MATCH_1}")
    if(OPENCSD_VERSION AND
       NOT OPENCSD_VERSION VERSION_LESS LLVM_OPENCSD_MIN_VERSION)
      set(OPENCSD_VERSION_OK ON)
    endif()
  endif()

  if(OPENCSD_LIB AND OPENCSD_INCLUDE AND OPENCSD_VERSION_OK)
    target_compile_definitions(LLVMProfileData PRIVATE HAVE_OPENCSD)
    target_include_directories(LLVMProfileData PRIVATE ${OPENCSD_INCLUDE})
    target_link_libraries(LLVMProfileData PRIVATE ${OPENCSD_LIB})
    message(STATUS "LLVMProfileData: OpenCSD support enabled (found ${OPENCSD_VERSION}).")
    set(LLVM_HAVE_OPENCSD ON CACHE BOOL "Is OpenCSD available" FORCE)
  elseif(OPENCSD_LIB AND OPENCSD_INCLUDE)
    if(LLVM_ENABLE_OPENCSD STREQUAL "FORCE_ON")
      message(FATAL_ERROR
        "OpenCSD found at ${OPENCSD_LIB} but its version "
        "(${OPENCSD_VERSION}) is older than the required "
        "${LLVM_OPENCSD_MIN_VERSION}. Install a newer OpenCSD or "
        "configure with -DLLVM_ENABLE_OPENCSD=OFF.")
    else()
      message(STATUS
        "LLVMProfileData: OpenCSD ${OPENCSD_VERSION} is older than "
        "${LLVM_OPENCSD_MIN_VERSION}; ETM decoding support disabled.")
    endif()
  else()
    if(LLVM_ENABLE_OPENCSD STREQUAL "FORCE_ON")
      message(FATAL_ERROR "OpenCSD enabled but library or headers not found.")
    else()
      message(STATUS "LLVMProfileData: OpenCSD not found; ETM decoding support disabled.")
    endif()
  endif()
endif()

add_subdirectory(Coverage)
