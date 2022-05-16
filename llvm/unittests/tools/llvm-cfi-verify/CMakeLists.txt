set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Object
  Support
  Symbolize
  )

add_llvm_unittest(CFIVerifyTests
  FileAnalysis.cpp
  GraphBuilder.cpp
  )
target_link_libraries(CFIVerifyTests PRIVATE LLVMCFIVerify)

set_property(TARGET CFIVerifyTests PROPERTY FOLDER "Tests/UnitTests/ToolTests")
