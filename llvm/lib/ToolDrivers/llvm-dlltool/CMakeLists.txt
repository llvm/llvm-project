set(LLVM_TARGET_DEFINITIONS Options.td)
llvm_tablegen(Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DllOptionsTableGen)

add_llvm_component_library(LLVMDlltoolDriver
  DlltoolDriver.cpp

  LINK_COMPONENTS
  BinaryFormat
  Object
  Option
  Support
  TargetParser
  )

add_dependencies(LLVMDlltoolDriver DllOptionsTableGen)
