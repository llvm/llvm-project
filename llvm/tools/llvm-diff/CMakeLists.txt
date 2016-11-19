set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-diff
  llvm-diff.cpp
  DiffConsumer.cpp
  DiffLog.cpp
  DifferenceEngine.cpp

  DEPENDS
  intrinsics_gen
  )
