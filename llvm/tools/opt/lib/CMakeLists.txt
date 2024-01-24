# We don't want to link this into libLLVM
add_llvm_library(LLVMOptDriver
  STATIC
  NewPMDriver.cpp
  optdriver.cpp
  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
)
