set(LLVM_LINK_COMPONENTS
  Core
  Support
  TransformUtils
  )

add_llvm_unittest(UtilsTests
  ASanStackFrameLayoutTest.cpp
  Cloning.cpp
  IntegerDivision.cpp
  Local.cpp
  MemorySSA.cpp
  ValueMapperTest.cpp
  )
