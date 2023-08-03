set(LLVM_LINK_COMPONENTS
  Support
  CAS
  RemoteCachingService
  )

add_llvm_tool(llvm-cas
  llvm-cas.cpp
  )
