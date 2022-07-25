if (LLVM_CAS_ENABLE_REMOTE_CACHE)
add_subdirectory(RemoteCacheProto)
add_subdirectory(RemoteCacheServer)
add_subdirectory(Client)
add_subdirectory(CAS)

set(LLVM_LINK_COMPONENTS
  RemoteCacheKVProto
  RemoteCacheCASProto
  RemoteCacheClient
  GRPCRelayCAS
  )
else ()
add_subdirectory(NullService)

set(LLVM_LINK_COMPONENTS
  RemoteNullService
  )
endif()

add_llvm_library(LLVMRemoteCachingService
  RemoteCachingService.cpp
)
