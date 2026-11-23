set(LLVM_LINK_COMPONENTS
  Demangle
  Support
)

add_llvm_unittest(DemangleTests
  DemangleTest.cpp
  DLangDemangleTest.cpp
  ItaniumDemangleTest.cpp
  MicrosoftDemangleTest.cpp
  OutputBufferTest.cpp
  PartialDemangleTest.cpp
  RustDemangleTest.cpp
)
