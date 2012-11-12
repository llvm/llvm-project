set(LLVM_LINK_COMPONENTS archive)

add_llvm_tool(llvm-ar
  llvm-ar.cpp
  )

# TODO: Support check-local.
