
set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  BinaryFormat
  DebugInfoDWARF
  DebugInfoPDB
  Core
  MC
  IPO
  MCDisassembler
  Object
  ProfileData
  Support
  Symbolize
  TargetParser
  )

add_llvm_tool(llvm-profgen
  llvm-profgen.cpp
  PerfReader.cpp
  CSPreInliner.cpp
  ProfiledBinary.cpp
  ProfileGenerator.cpp
  MissingFrameInferrer.cpp
  )

set(LLVM_ENABLE_OPENCSD "AUTO" CACHE STRING "Enable OpenCSD support in llvm-profgen")

if(NOT LLVM_ENABLE_OPENCSD STREQUAL "OFF")
  find_library(OPENCSD_LIB NAMES opencsd_c_api opencsd)
  find_path(OPENCSD_INCLUDE NAMES opencsd/ocsd_if_types.h)

  if(OPENCSD_LIB AND OPENCSD_INCLUDE)
    target_compile_definitions(llvm-profgen PRIVATE HAVE_OPENCSD)
    target_include_directories(llvm-profgen PRIVATE ${OPENCSD_INCLUDE})
    target_link_libraries(llvm-profgen PRIVATE ${OPENCSD_LIB})
    message(STATUS "llvm-profgen: OpenCSD support enabled.")
  else()
    if(LLVM_ENABLE_OPENCSD STREQUAL "ON")
      message(FATAL_ERROR "OpenCSD enabled but library or headers not found.")
    else()
      message(STATUS "llvm-profgen: OpenCSD not found; ETM decoding support disabled.")
    endif()
  endif()
endif()
