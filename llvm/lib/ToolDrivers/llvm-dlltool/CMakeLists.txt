set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DllOptionsTableGen)

add_llvm_library(LLVMDlltoolDriver
  DlltoolDriver.cpp
  )

add_dependencies(LLVMDlltoolDriver DllOptionsTableGen)
