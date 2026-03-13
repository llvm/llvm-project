add_llvm_unittest(SupportHTTPTests
  HTTPServerTests.cpp
  )

target_link_libraries(SupportHTTPTests PRIVATE
  LLVMSupportHTTP
  LLVMTestingSupport
  )
