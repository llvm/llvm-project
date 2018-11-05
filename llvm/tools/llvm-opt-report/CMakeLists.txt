set(LLVM_LINK_COMPONENTS Core Demangle Object OptRemarks Support)

add_llvm_tool(llvm-opt-report
  OptReport.cpp
  )

