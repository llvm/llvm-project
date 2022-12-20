set(LLVM_LINK_COMPONENTS
  BitWriter
  Core
  IPO
  IRReader
  Passes
  Support
  )

add_llvm_tool(llvm-extract
  llvm-extract.cpp

  DEPENDS
  intrinsics_gen
  )
