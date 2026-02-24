set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  IRPrinter
  IRReader
  Support
  )

add_llvm_tool(llubi
  llubi.cpp

  DEPENDS
  intrinsics_gen
  )

add_subdirectory(lib)
target_link_libraries(llubi PRIVATE LLVMUBAwareInterpreter)
