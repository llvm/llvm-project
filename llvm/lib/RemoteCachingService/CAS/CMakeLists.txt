add_llvm_library(LLVMGRPCRelayCAS
  GRPCRelayCAS.cpp

  LINK_COMPONENTS
  RemoteCacheKVProto
  RemoteCacheCASProto
  RemoteCacheClient
  CAS
)
