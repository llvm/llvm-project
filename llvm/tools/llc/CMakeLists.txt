# Add GlobalISel to the dependencies if the user wants to build it.
if(LLVM_BUILD_GLOBAL_ISEL)
  set(GLOBAL_ISEL GlobalISel)
else()
  set(GLOBAL_ISEL "")
endif()

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  Analysis
  AsmPrinter
  CodeGen
  Core
  IRReader
  ${GLOBAL_ISEL}
  MC
  MIRParser
  ScalarOpts
  SelectionDAG
  Support
  Target
  TransformUtils
  )

# Support plugins.
set(LLVM_NO_DEAD_STRIP 1)

add_llvm_tool(llc
  llc.cpp
  )
export_executable_symbols(llc)
