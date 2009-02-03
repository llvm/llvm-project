set(LLVM_LINK_COMPONENTS archive)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

# TODO: Support check-local.
