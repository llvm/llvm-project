set(LLVM_LINK_COMPONENTS jit bitwriter nativecodegen interpreter)

add_llvm_example(BrainF
  BrainF.cpp
  BrainFDriver.cpp
  )
