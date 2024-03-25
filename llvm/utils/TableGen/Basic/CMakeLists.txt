# The basic TableGen library contains as little dependencies as possible.
# In particular, it does not depend on vt_gen -> it does not use ValueTypes.
#
# This library is the only thing included in `llvm-min-tablegen`.

set(LLVM_LINK_COMPONENTS
  Support
  TableGen
  )

add_llvm_library(LLVMTableGenBasic STATIC OBJECT EXCLUDE_FROM_ALL
  CodeGenIntrinsics.cpp
  SDNodeProperties.cpp
)
set_target_properties(LLVMTableGenBasic PROPERTIES FOLDER "Tablegenning")

# Users may include its headers as "Basic/*.h"
target_include_directories(LLVMTableGenBasic
  INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
  )
