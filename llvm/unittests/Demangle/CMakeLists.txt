set(DemangleSources
  PartialDemangleTest.cpp
)

add_llvm_unittest(DemangleTests
  ${DemangleSources}
)
