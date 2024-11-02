set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CodeGen
  CodeGenTypes
  Core
  FileCheck
  MC
  MIRParser
  Passes
  Support
  Target
  TargetParser
  )

add_llvm_unittest(MIRTests
  MachineMetadata.cpp
  PassBuilderCallbacksTest.cpp
  )

target_link_libraries(MIRTests PRIVATE LLVMTestingSupport)
