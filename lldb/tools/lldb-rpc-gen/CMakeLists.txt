add_lldb_tool(lldb-rpc-gen
    RPCCommon.cpp
    server/RPCServerHeaderEmitter.cpp
    server/RPCServerSourceEmitter.cpp
    lldb-rpc-gen.cpp

    CLANG_LIBS
      clangAST
      clangBasic
      clangCodeGen
      clangFrontend
      clangLex
      clangRewrite
      clangSerialization
      clangTooling

    LINK_COMPONENTS
      Support
  )

if (NOT DEFINED LLDB_RPC_GEN_EXE)
  set(LLDB_RPC_GEN_EXE $<TARGET_FILE:lldb-rpc-gen> CACHE STRING "Executable that generates lldb-rpc-server")
endif()
