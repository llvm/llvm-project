set(LLVM_LINK_COMPONENTS
  Demangle
  Remarks
  Support
  )

add_llvm_tool(llvm-remark-size-diff
  RemarkSizeDiff.cpp
  )
