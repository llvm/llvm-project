add_llvm_library(LLVMLTO
  Caching.cpp
  LTO.cpp
  LTOBackend.cpp
  LTOModule.cpp
  LTOCodeGenerator.cpp
  UpdateCompilerUsed.cpp
  ThinLTOCodeGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LTO

  DEPENDS
  intrinsics_gen
  llvm_vcsrevision_h
)
