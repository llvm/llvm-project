set(LLVM_LINK_COMPONENTS
  Demangle
  Support
)

add_llvm_unittest(DemangleTests
  ItaniumDemangleTest.cpp
  PartialDemangleTest.cpp
)
