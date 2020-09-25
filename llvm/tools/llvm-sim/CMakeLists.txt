set(LLVM_LINK_COMPONENTS core support object coverage AsmParser IRReader TransformUtils ipo)

add_llvm_tool(llvm-sim
  llvm-sim.cpp
)
