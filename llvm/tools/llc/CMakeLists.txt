set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  Analysis
  AsmPrinter
  CodeGen
  CodeGenTypes
  Core
  IPO
  IRPrinter
  IRReader
  MC
  MIRParser
  Passes
  Plugins
  Remarks
  ScalarOpts
  SelectionDAG
  Support
  Target
  TargetParser
  TransformUtils
  Vectorize
  )

# Keep the command-line implementation in a separate library so other LLVM
# tools can reuse the llc driver without compiling or rewriting llc.cpp.
add_llvm_library(LLVMLlcDriver
  STATIC
  llcdriver.cpp
  NewPMDriver.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
  )

add_llvm_tool(llc
  PARTIAL_SOURCES_INTENDED
  llc.cpp

  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(llc PRIVATE LLVMLlcDriver)

export_executable_symbols_for_plugins(llc)
