set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  Support
  TransformUtils
  )

add_llvm_unittest(UtilsTests
  ASanStackFrameLayoutTest.cpp
  Cloning.cpp
  FunctionComparator.cpp
  IntegerDivision.cpp
  Local.cpp
  MemorySSA.cpp
  ValueMapperTest.cpp
  )
