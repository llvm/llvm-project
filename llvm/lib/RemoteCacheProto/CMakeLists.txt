include(FindGRPC)

generate_protos(llvmRemoteCacheKVProto "compilation_caching_kv.proto"
  GRPC)
generate_protos(llvmRemoteCacheCASProto "compilation_caching_cas.proto"
  GRPC)
