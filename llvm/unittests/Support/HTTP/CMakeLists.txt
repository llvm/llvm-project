set(LLVM_LINK_COMPONENTS
  LLVMSupportHTTP
  )

add_llvm_unittest(SupportHTTPTests
  HTTPServerTests.cpp
  )

target_link_libraries(SupportHTTPTests PRIVATE
  LLVMTestingSupport
  )
