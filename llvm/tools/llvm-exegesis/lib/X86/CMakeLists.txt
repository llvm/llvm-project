include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/X86
  ${LLVM_BINARY_DIR}/lib/Target/X86
  )

add_llvm_library(LLVMExegesisX86
  Target.cpp

  STATIC

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  X86
  Exegesis
  )
