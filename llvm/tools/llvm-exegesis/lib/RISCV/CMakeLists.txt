include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
  )

set(LLVM_LINK_COMPONENTS
  RISCV
  CodeGenTypes
  Core
  Exegesis
  MC
  Support
  )

add_llvm_library(LLVMExegesisRISCV
  DISABLE_LLVM_LINK_LLVM_DYLIB
  STATIC
  RISCVExegesisPostprocessing.cpp
  RISCVExegesisPreprocessing.cpp
  Target.cpp

  DEPENDS
  intrinsics_gen
  RISCVCommonTableGen
  )
