add_llvm_library(LLVMVectorize
  LoadStoreVectorizer.cpp
  LoopVectorize.cpp
  SLPVectorizer.cpp
  Vectorize.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Transforms

  DEPENDS
  intrinsics_gen
  )
