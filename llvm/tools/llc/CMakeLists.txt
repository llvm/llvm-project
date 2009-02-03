set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} bitreader)

add_llvm_tool(llc
  llc.cpp
  )
