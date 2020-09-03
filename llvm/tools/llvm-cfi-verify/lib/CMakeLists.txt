add_library(LLVMCFIVerify
  STATIC
  FileAnalysis.cpp
  FileAnalysis.h
  GraphBuilder.cpp
  GraphBuilder.h
  )

llvm_update_compile_flags(LLVMCFIVerify)
if (LLVM_LINK_LLVM_DYLIB)
  set(libs LLVM)
else()
  llvm_map_components_to_libnames(libs
    DebugInfoDWARF
    MC
    MCParser
    Object
    Support
    Symbolize
    )
endif()
target_link_libraries(LLVMCFIVerify ${libs})
set_target_properties(LLVMCFIVerify PROPERTIES FOLDER "Libraries")
