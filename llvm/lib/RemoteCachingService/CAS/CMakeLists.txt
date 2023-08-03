add_llvm_component_library(LLVMGRPCRelayCAS
  GRPCRelayCAS.cpp

  LINK_COMPONENTS
  RemoteCacheKVProto
  RemoteCacheCASProto
  RemoteCacheClient
  Support
  CAS
)
