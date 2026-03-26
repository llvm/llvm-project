add_llvm_unittest(SupportHTTPTests
  HTTPServerTests.cpp
  )

if(LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(SupportHTTPTests PRIVATE
    LLVM
    )
else()
  target_link_libraries(SupportHTTPTests PRIVATE
    LLVMSupportHTTP
    LLVMTestingSupport
    )
endif()
