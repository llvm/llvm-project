set(LLVM_LINK_COMPONENTS
  Demangle
  )

add_llvm_tool(llvm-cxxfilt
  llvm-cxxfilt.cpp
  )
