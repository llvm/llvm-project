set(LLVM_LINK_COMPONENTS Core Demangle Object Support)

add_llvm_tool(llvm-opt-report
  OptReport.cpp
  )

