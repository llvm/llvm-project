add_llvm_library(ReferencePlugin SHARED INSTALL_WITH_TOOLCHAIN
  ReferencePlugin.cpp

  DEPENDS
  intrinsics_gen
)
