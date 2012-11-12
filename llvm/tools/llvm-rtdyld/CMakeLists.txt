set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} support MC object RuntimeDyld JIT)

add_llvm_tool(llvm-rtdyld
  llvm-rtdyld.cpp
  )
