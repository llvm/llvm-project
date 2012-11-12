set(LLVM_LINK_COMPONENTS ${LLVM_TARGETS_TO_BUILD} bitreader asmparser bitwriter instrumentation scalaropts ipo vectorize)

add_llvm_tool(opt
  AnalysisWrappers.cpp
  GraphPrinters.cpp
  PrintSCC.cpp
  opt.cpp
  )
