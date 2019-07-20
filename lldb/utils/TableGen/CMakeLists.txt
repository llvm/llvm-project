# tablegen targets get exported via llvm for LLVMConfig.cmake. So standalone
# builds of lldb can potentially import this via LLVMConfig and also attempt to
# build it in tree. So only build it if it doesn't exist.
if (TARGET lldb-tblgen)
  set(LLDB_TABLEGEN_EXE $<TARGET_FILE:lldb-tblgen> CACHE STRING "")
else()
  set(LLVM_LINK_COMPONENTS Support)

  add_tablegen(lldb-tblgen LLDB
    LLDBOptionDefEmitter.cpp
    LLDBTableGen.cpp
    )
  set_target_properties(lldb-tblgen PROPERTIES FOLDER "LLDB tablegenning")
endif()
