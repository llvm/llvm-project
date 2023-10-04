set(LLVM_LINK_COMPONENTS
  Support
  CAS
  RemoteCachingService
  CASUtil
  )

add_llvm_tool(llvm-cas
  llvm-cas.cpp
  )
