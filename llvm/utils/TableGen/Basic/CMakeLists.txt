# The basic TableGen library contains as little dependencies as possible.
# In particular, it does not depend on vt_gen -> it does not use ValueTypes.
#
# This library is the only thing included in `llvm-min-tablegen`.

set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenBasic OBJECT EXCLUDE_FROM_ALL DISABLE_LLVM_LINK_LLVM_DYLIB
  ARMTargetDefEmitter.cpp
  Attributes.cpp
  CodeGenIntrinsics.cpp
  DirectiveEmitter.cpp
  IntrinsicEmitter.cpp
  RISCVTargetDefEmitter.cpp
  SDNodeProperties.cpp
  TableGen.cpp
  VTEmitter.cpp
)

# Users may include its headers as "Basic/*.h"
target_include_directories(LLVMTableGenBasic
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  )
