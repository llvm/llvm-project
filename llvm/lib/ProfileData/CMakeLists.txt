add_llvm_component_library(LLVMProfileData
  GCOV.cpp
  InstrProf.cpp
  InstrProfReader.cpp
  InstrProfWriter.cpp
  ProfileSummaryBuilder.cpp
  SampleProf.cpp
  SampleProfReader.cpp
  SampleProfWriter.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/ProfileData

  DEPENDS
  intrinsics_gen
  )

add_subdirectory(Coverage)
