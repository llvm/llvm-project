set(LLVM_LINK_COMPONENTS asmparser instrumentation scalaropts ipo
  linker bitreader bitwriter)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(bugpoint
  BugDriver.cpp
  CrashDebugger.cpp
  ExecutionDriver.cpp
  ExtractFunction.cpp
  FindBugs.cpp
  Miscompilation.cpp
  OptimizerDriver.cpp
  TestPasses.cpp
  ToolRunner.cpp
  bugpoint.cpp
  )
