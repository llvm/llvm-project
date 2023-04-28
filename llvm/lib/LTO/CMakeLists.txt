if (LLVM_CAS_ENABLE_REMOTE_CACHE)
  set(LLVM_LINK_COMPONENTS RemoteCacheClient)
else()
  set(LLVM_LINK_COMPONENTS RemoteNullService)
endif()

add_llvm_component_library(LLVMLTO
  LTO.cpp
  LTOBackend.cpp
  LTOModule.cpp
  LTOCodeGenerator.cpp
  SummaryBasedOptimizations.cpp
  UpdateCompilerUsed.cpp
  ThinLTOCodeGenerator.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LTO
  DEPENDS
  intrinsics_gen
  llvm_vcsrevision_h

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  BinaryFormat
  BitReader
  BitWriter
  CAS
  CodeGen
  Core
  Extensions
  IPO
  InstCombine
  Instrumentation
  Linker
  MC
  ObjCARC
  Object
  Passes
  Remarks
  RemoteCachingService
  Scalar
  Support
  Target
  TargetParser
  TransformUtils
  )
