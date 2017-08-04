set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  Passes
  )

set(IRSources
  AsmWriterTest.cpp
  AttributesTest.cpp
  BasicBlockTest.cpp
  CFGBuilder.cpp
  ConstantRangeTest.cpp
  ConstantsTest.cpp
  DebugInfoTest.cpp
  DebugTypeODRUniquingTest.cpp
  DominatorTreeTest.cpp
  FunctionTest.cpp
  PassBuilderCallbacksTest.cpp
  IRBuilderTest.cpp
  InstructionsTest.cpp
  IntrinsicsTest.cpp
  LegacyPassManagerTest.cpp
  MDBuilderTest.cpp
  MetadataTest.cpp
  ModuleTest.cpp
  PassManagerTest.cpp
  PatternMatch.cpp
  TypeBuilderTest.cpp
  TypesTest.cpp
  UseTest.cpp
  UserTest.cpp
  ValueHandleTest.cpp
  ValueMapTest.cpp
  ValueTest.cpp
  VerifierTest.cpp
  WaymarkTest.cpp
  )

# HACK: Declare a couple of source files as optionally compiled to satisfy the
# missing-file-checker in LLVM's weird CMake build.
set(LLVM_OPTIONAL_SOURCES
  ValueMapTest.cpp
  )

add_llvm_unittest(IRTests
  ${IRSources}
  )
