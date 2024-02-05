set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenGlobalISel STATIC DISABLE_LLVM_LINK_LLVM_DYLIB
  CodeExpander.cpp
  CombinerUtils.cpp
  CXXPredicates.cpp
  MatchDataInfo.cpp
  PatternParser.cpp
  Patterns.cpp
  $<TARGET_OBJECTS:obj.LLVMTableGenCommon>

  DEPENDS
  vt_gen
  )

# Users may include its headers as "GlobalISel/*.h"
target_include_directories(LLVMTableGenGlobalISel
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  )
