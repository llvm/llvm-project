set(LLVM_LINK_COMPONENTS
  RemoteCacheKVProto
  RemoteCacheCASProto
  )

include_directories($<TARGET_PROPERTY:LLVMRemoteCacheKVProto,BINARY_DIR>)
include_directories($<TARGET_PROPERTY:LLVMRemoteCacheCASProto,BINARY_DIR>)

add_llvm_component_library(LLVMRemoteCacheServer
  LLVMCASCacheProvider.cpp
  RemoteCacheServer.cpp

  LINK_COMPONENTS
  Support
  CAS
)
