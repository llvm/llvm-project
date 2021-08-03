set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-livepatch
  llvm-livepatch.cpp

  DEPENDS
  intrinsics_gen
  )
