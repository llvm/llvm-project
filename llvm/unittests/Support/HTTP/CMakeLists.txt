set(LLVM_LINK_COMPONENTS
  SupportHTTP
  )

add_llvm_unittest(SupportHTTPTests
  HTTPServerTests.cpp
  )

target_link_libraries(SupportHTTPTests PRIVATE
  LLVMTestingSupport
  )
