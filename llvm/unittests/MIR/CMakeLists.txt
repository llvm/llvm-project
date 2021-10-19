set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  CodeGen
  Core
  FileCheck
  MC
  MIRParser
  Support
  Target
  )

add_llvm_unittest(MIRTests
  MachineMetadata.cpp
  )

target_link_libraries(MIRTests PRIVATE LLVMTestingSupport)
