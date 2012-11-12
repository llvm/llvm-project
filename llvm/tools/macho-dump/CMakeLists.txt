set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} support object)

add_llvm_tool(macho-dump
  macho-dump.cpp
  )
