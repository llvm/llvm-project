set(LLVM_REQUIRES_EH 1)
set(LLVM_LINK_COMPONENTS bitreader bitwriter instrumentation scalaropts ipo)

add_llvm_tool(opt
  AnalysisWrappers.cpp
  GraphPrinters.cpp
  PrintSCC.cpp
  opt.cpp
  )
