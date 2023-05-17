set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Coroutines
  Passes
  Support
  TargetParser
  TransformUtils
  )

add_llvm_unittest(CoroTests
  ExtraRematTest.cpp
  )

target_link_libraries(CoroTests PRIVATE LLVMTestingSupport)

set_property(TARGET CoroTests PROPERTY FOLDER "Tests/UnitTests/TransformTests")
