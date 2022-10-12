function(generate_llvm_protos_library LibraryName ProtoFile)
  # Take the first two args and forward the remaining to generate_proto_sources.
  cmake_parse_arguments(PARSE_ARGV 2 PROTO "" "" "")
  generate_proto_sources(ProtoSource ${ProtoFile} ${PROTO_UNPARSED_ARGUMENTS})

  add_llvm_library(${LibraryName} ${ProtoSource}
    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC grpc++ gpr protobuf)
endfunction()

generate_llvm_protos_library(LLVMRemoteCacheKVProto "compilation_caching_kv.proto" GRPC)
generate_llvm_protos_library(LLVMRemoteCacheCASProto "compilation_caching_cas.proto" GRPC)

