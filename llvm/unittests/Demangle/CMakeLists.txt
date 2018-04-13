set(LLVM_LINK_COMPONENTS
  Demangle
)

set(DemangleSources
  PartialDemangleTest.cpp
)

add_llvm_unittest(DemangleTests
  ${DemangleSources}
)
