set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Instrumentation
  Passes
  ProfileData
  Support
)

add_llvm_unittest(InstrumentationTests
  MemProfUseTest.cpp
  PGOInstrumentationTest.cpp
  )

target_link_libraries(InstrumentationTests PRIVATE LLVMTestingSupport)

set_property(TARGET InstrumentationTests PROPERTY FOLDER "Tests/UnitTests/TransformTests")
