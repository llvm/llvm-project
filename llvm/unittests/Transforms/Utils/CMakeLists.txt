set(LLVM_LINK_COMPONENTS
  TransformUtils
  )

add_llvm_unittest(UtilsTests
  Cloning.cpp
  IntegerDivision.cpp
  Local.cpp
  )
