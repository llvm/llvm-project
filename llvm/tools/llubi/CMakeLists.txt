set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  IRReader
  IRPrinter
  Support
  )

add_llvm_tool(llubi
  llubi.cpp
  Context.cpp
  Interpreter.cpp
  Value.cpp

  DEPENDS
  intrinsics_gen
  )
