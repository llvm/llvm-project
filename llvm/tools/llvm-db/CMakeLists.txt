set(LLVM_LINK_COMPONENTS debugger)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-db
  CLIDebugger.cpp
  Commands.cpp
  llvm-db.cpp
  )
