set(LLVM_PRIVATE_LINK_COMPONENTS
    DebugInfoPDB)

add_lldb_library(lldbPluginSymbolFilePDB
  SymbolFilePDB.cpp
  )
