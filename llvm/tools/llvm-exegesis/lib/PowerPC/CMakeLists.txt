include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/PowerPC
  ${LLVM_LIBRARY_DIR}/Target/PowerPC
  )

set(LLVM_LINK_COMPONENTS
  PowerPC
  Exegesis
  Core
  Support
  )

add_llvm_library(LLVMExegesisPowerPC
  DISABLE_LLVM_LINK_LLVM_DYLIB
  STATIC
  Target.cpp

  DEPENDS
  intrinsics_gen
  PowerPCCommonTableGen
  )
