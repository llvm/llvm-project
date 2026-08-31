add_llvm_unittest(HTTPTests
  HTTPServerTests.cpp
  )

target_link_libraries(HTTPTests PRIVATE
  LLVMHTTP
  LLVMTestingSupport
  )
