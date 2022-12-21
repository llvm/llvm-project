include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  )

set(LLVM_LINK_COMPONENTS
  X86
  Exegesis
  Core
  Support
  TargetParser
  CodeGen
  MC
  )

add_llvm_library(LLVMExegesisX86
  DISABLE_LLVM_LINK_LLVM_DYLIB
  STATIC
  Target.cpp
  X86Counter.cpp

  DEPENDS
  intrinsics_gen
  X86CommonTableGen
  )
