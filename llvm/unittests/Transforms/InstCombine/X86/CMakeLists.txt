set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Passes
  Support
  ScalarOpts
  TransformUtils
  InstCombine
  Target
  TargetParser
  X86CodeGen
  X86Desc
  X86Info
  MC
  )

add_llvm_unittest(InstCombineX86Tests
  TernlogTest.cpp
  )

target_link_libraries(InstCombineX86Tests PRIVATE LLVMTestingSupport)
set_property(TARGET InstCombineX86Tests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")
