set(LLVM_LINK_COMPONENTS support system)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvmc
  Action.cpp
  CompilationGraph.cpp
  llvmc.cpp
  Plugin.cpp
  Tool.cpp
  )
