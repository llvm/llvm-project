# This file comes from https://reviews.llvm.org/D78978.
# Author: [@paolosev](https://reviews.llvm.org/p/paolosev/).

add_lldb_library(lldbPluginProcessWasm PLUGIN
  ProcessWasm.cpp
  ThreadWasm.cpp
  UnwindWasm.cpp
  wasmRegisterContext.cpp

  LINK_LIBS
    lldbCore
    ${LLDB_PLUGINS}
  LINK_COMPONENTS
    Support
  )
