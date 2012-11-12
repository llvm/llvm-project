set(LLVM_LINK_COMPONENTS asmparser instrumentation scalaropts ipo
  linker bitreader bitwriter vectorize)

add_llvm_tool(bugpoint
  BugDriver.cpp
  CrashDebugger.cpp
  ExecutionDriver.cpp
  ExtractFunction.cpp
  FindBugs.cpp
  Miscompilation.cpp
  OptimizerDriver.cpp
  ToolRunner.cpp
  bugpoint.cpp
  )
