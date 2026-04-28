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

set(LLVM_ENABLE_OPENCSD "AUTO" CACHE STRING "Enable OpenCSD support in LLVMProfileData")

if(NOT LLVM_ENABLE_OPENCSD STREQUAL "OFF")
  find_library(OPENCSD_LIB NAMES opencsd_c_api opencsd)
  find_path(OPENCSD_INCLUDE NAMES opencsd/ocsd_if_types.h)

  if(OPENCSD_LIB AND OPENCSD_INCLUDE)
    target_compile_definitions(LLVMProfileData PRIVATE HAVE_OPENCSD)
    target_include_directories(LLVMProfileData PRIVATE ${OPENCSD_INCLUDE})
    target_link_libraries(LLVMProfileData PRIVATE ${OPENCSD_LIB})
    message(STATUS "LLVMProfileData: OpenCSD support enabled.")
  else()
    if(LLVM_ENABLE_OPENCSD STREQUAL "ON")
      message(FATAL_ERROR "OpenCSD enabled but library or headers not found.")
    else()
      message(STATUS "LLVMProfileData: OpenCSD not found; ETM decoding support disabled.")
    endif()
  endif()
endif()

add_subdirectory(Coverage)
