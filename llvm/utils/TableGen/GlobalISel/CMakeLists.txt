set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenGlobalISel STATIC OBJECT EXCLUDE_FROM_ALL
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
