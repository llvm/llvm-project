set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  UncoalescedAnalysis
  BlockSizeInvarianceAnalysis
  IRReader
  Support
  )

add_llvm_tool(llvm-gpudrano
  llvm-gpudrano.cpp

  DEPENDS
  intrinsics_gen
  )
