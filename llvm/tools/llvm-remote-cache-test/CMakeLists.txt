include_directories($<TARGET_PROPERTY:llvmRemoteCacheKVProto,BINARY_DIR>)
include_directories($<TARGET_PROPERTY:llvmRemoteCacheCASProto,BINARY_DIR>)

set(LLVM_LINK_COMPONENTS
  CAS
  RemoteCacheCASProto
  RemoteCacheKVProto
  Support
)

add_llvm_tool(llvm-remote-cache-test
  llvm-remote-cache-test.cpp
  LLVMCASCacheProvider.cpp
  RemoteCacheServer.cpp
)
