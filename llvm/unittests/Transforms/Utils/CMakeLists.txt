set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BitWriter
  Core
  Support
  TransformUtils
  Passes
  Vectorize
  )

add_llvm_unittest(UtilsTests
  ASanStackFrameLayoutTest.cpp
  BasicBlockUtilsTest.cpp
  CallPromotionUtilsTest.cpp
  CloningTest.cpp
  CodeExtractorTest.cpp
  CodeLayoutTest.cpp
  CodeMoverUtilsTest.cpp
  DebugifyTest.cpp
  FunctionComparatorTest.cpp
  IntegerDivisionTest.cpp
  LocalTest.cpp
  LoopRotationUtilsTest.cpp
  LoopUtilsTest.cpp
  MemTransferLowering.cpp
  ModuleUtilsTest.cpp
  ScalarEvolutionExpanderTest.cpp
  SizeOptsTest.cpp
  SSAUpdaterBulkTest.cpp
  UnrollLoopTest.cpp
  ValueMapperTest.cpp
  ProfDataUtilTest.cpp
  )

set_property(TARGET UtilsTests PROPERTY FOLDER "Tests/UnitTests/TransformsTests")
