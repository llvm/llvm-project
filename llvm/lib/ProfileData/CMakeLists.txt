add_llvm_component_library(LLVMProfileData
  DataAccessProf.cpp
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

# InstrProfReader.cpp and InstrProfWriter.cpp have ambiguous 'Summary' due
# to namespace conflicts in unity builds.
set_source_files_properties(
  InstrProfReader.cpp
  InstrProfWriter.cpp
  SampleProf.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)

add_subdirectory(Coverage)

# 'using namespace sampleprof' causes ambiguity with other files.
set_source_files_properties(
  SampleProfReader.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
