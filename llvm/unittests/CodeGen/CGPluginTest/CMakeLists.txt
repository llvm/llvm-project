set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Support
  Core
  MC
  Target
  TargetParser
  CodeGen
  )

add_llvm_unittest(CGPluginTest
  PluginTest.cpp
  Plugin/CodeGenTestPass.cpp

  EXPORT_SYMBOLS
  )
target_link_libraries(CGPluginTest PRIVATE LLVMTestingSupport)

unset(LLVM_LINK_COMPONENTS)
add_subdirectory(Plugin)
