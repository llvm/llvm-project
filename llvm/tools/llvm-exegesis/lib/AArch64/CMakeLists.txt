include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AArch64
  ${LLVM_BINARY_DIR}/lib/Target/AArch64
  )

add_llvm_library(LLVMExegesisAArch64
  Target.cpp

  STATIC

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  AArch64
  Exegesis
  )
