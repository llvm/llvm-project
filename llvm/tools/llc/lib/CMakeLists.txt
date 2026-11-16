# Keep the command-line implementation in a separate library so other LLVM
# tools can reuse the llc driver without compiling or rewriting llc.cpp.
add_llvm_library(LLVMLlcDriver
  STATIC
  llcdriver.cpp
  NewPMDriver.cpp

  DEPENDS
  intrinsics_gen
  )
