set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  Core
  Support
  TransformUtils
  )

add_llvm_unittest(UtilsTests
  ASanStackFrameLayoutTest.cpp
  Cloning.cpp
  CodeExtractor.cpp
  FunctionComparator.cpp
  IntegerDivision.cpp
  Local.cpp
  OrderedInstructions.cpp
  ValueMapperTest.cpp
  )
