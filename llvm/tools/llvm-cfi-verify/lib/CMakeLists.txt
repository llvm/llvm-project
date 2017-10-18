add_library(LLVMCFIVerify
  STATIC
  FileAnalysis.cpp
  FileAnalysis.h)

llvm_update_compile_flags(LLVMCFIVerify)
llvm_map_components_to_libnames(libs
  MC
  MCParser
  Object
  Support)
target_link_libraries(LLVMCFIVerify ${libs})
