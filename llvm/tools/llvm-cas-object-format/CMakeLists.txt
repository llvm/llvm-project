set(LLVM_LINK_COMPONENTS
  Support
  CAS
  RemoteCachingService
  MCCAS
  )

add_llvm_tool(llvm-cas-object-format
  llvm-cas-object-format.cpp
  )