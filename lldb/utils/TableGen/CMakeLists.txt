set(LLVM_LINK_COMPONENTS Support)

add_tablegen(lldb-tblgen LLDB
  LLDBOptionDefEmitter.cpp
  LLDBTableGen.cpp
  )
set_target_properties(lldb-tblgen PROPERTIES FOLDER "LLDB tablegenning")

