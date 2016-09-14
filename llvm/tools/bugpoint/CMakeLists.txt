set(LLVM_LINK_COMPONENTS
  Analysis
  BitWriter
  CodeGen
  Core
  IPO
  IRReader
  InstCombine
  Instrumentation
  Linker
  ObjCARCOpts
  ScalarOpts
  Support
  Target
  TransformUtils
  Vectorize
  )

# Support plugins.
set(LLVM_NO_DEAD_STRIP 1)

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
export_executable_symbols(bugpoint)

if(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
  target_link_libraries(bugpoint Polly)
  if(POLLY_LINK_LIBS)
    foreach(lib ${POLLY_LINK_LIBS})
      target_link_libraries(bugpoint ${lib})
    endforeach(lib)
  endif(POLLY_LINK_LIBS)
  # Ensure LLVMTarget can resolve dependences in Polly.
  target_link_libraries(bugpoint LLVMTarget)
endif(WITH_POLLY AND LINK_POLLY_INTO_TOOLS)
