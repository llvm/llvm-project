add_llvm_component_library(LLVMLTO
  LTOBackend.cpp
  LTOModule.cpp
  LTOCodeGenerator.cpp
  UpdateCompilerUsed.cpp
  ThinLTOCodeGenerator.cpp
  # warning: attribute declaration must precede definition [-Wignored-attributes]
  # LLVM_ABI extern cl::opt<bool> EnableLTOInternalization
  LTO.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/LTO
  DEPENDS
  intrinsics_gen
  llvm_vcsrevision_h

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  AggressiveInstCombine
  Analysis
  BinaryFormat
  BitReader
  BitWriter
  CGData
  CodeGen
  CodeGenTypes
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
  Plugins
  Remarks
  Scalar
  Support
  Target
  TargetParser
  TransformUtils
  )
