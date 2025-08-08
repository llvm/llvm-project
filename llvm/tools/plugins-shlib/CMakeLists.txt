add_llvm_library(ReferencePlugin SHARED INSTALL_WITH_TOOLCHAIN
  Instrumentor.cpp
  ReferencePlugin.cpp

  DEPENDS
  intrinsics_gen
)
