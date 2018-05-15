set(LLVM_LINK_COMPONENTS
  Demangle
)

add_llvm_unittest(DemangleTests
  PartialDemangleTest.cpp
)
