set(LLVM_LINK_COMPONENTS
  CAS
  RemoteCacheServer
  Support
)

add_llvm_tool(llvm-remote-cache-test
  llvm-remote-cache-test.cpp
)
