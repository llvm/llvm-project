set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenGlobalISel STATIC DISABLE_LLVM_LINK_LLVM_DYLIB
  CodeExpander.cpp
  CXXPredicates.cpp
  MatchDataInfo.cpp
  Patterns.cpp

  DEPENDS
  vt_gen
  )

# Users may include its headers as "GlobalISel/*.h"
target_include_directories(LLVMTableGenGlobalISel
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  )
