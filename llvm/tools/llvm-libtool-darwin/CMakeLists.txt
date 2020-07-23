set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(llvm-libtool-darwin
  llvm-libtool-darwin.cpp
)
