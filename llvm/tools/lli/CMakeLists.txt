set(LLVM_LINK_COMPONENTS jit interpreter nativecodegen bitreader selectiondag)

add_llvm_tool(lli
  lli.cpp
  )
