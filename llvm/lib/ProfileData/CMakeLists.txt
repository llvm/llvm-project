add_llvm_component_library(LLVMProfileData
  GCOV.cpp
  InstrProf.cpp
  InstrProfCorrelator.cpp
  InstrProfReader.cpp
  InstrProfWriter.cpp
  ProfileSummaryBuilder.cpp
  SampleProf.cpp
  SampleProfReader.cpp
  SampleProfWriter.cpp
  RawMemProfReader.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ProfileData

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Core
  Support
  Demangle
  Object
  DebugInfoDWARF
  )

add_subdirectory(Coverage)
