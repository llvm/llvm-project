include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/RISCV
  ${LLVM_BINARY_DIR}/lib/Target/RISCV
)

set(LLVM_LINK_COMPONENTS
  CodeGen
  RISCV
  Exegesis
  Core
  Support
  )

add_llvm_library(LLVMExegesisRISCV
  DISABLE_LLVM_LINK_LLVM_DYLIB
  STATIC
  Target.cpp

  DEPENDS
  intrinsics_gen
  RISCVCommonTableGen
  )
