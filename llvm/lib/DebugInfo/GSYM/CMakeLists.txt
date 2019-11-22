add_llvm_component_library(LLVMDebugInfoGSYM
  Header.cpp
  FileWriter.cpp
  FunctionInfo.cpp
  GsymCreator.cpp
  GsymReader.cpp
  InlineInfo.cpp
  LineTable.cpp
  Range.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo/GSYM
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/DebugInfo

  DEPENDS
  LLVMMC
  )
