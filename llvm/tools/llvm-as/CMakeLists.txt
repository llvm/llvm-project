set(LLVM_LINK_COMPONENTS
  AsmParser
  BitWriter
  Core
  Support
  )

add_llvm_tool(llvm-as
  llvm-as.cpp

  DEPENDS
  intrinsics_gen
  )

setup_host_tool(llvm-as LLVM_AS llvm_as_exe llvm_as_target)
