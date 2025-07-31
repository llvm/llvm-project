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

if (CMAKE_CROSSCOMPILING)
  setup_host_tool(lldb-rpc-gen LLDB_RPC_GEN_EXE lldb_rpc_gen_exe lldb_rpc_gen_target)
else()
  if (NOT DEFINED LLDB_RPC_GEN_EXE)
    set(LLDB_RPC_GEN_EXE $<TARGET_FILE:lldb-rpc-gen> CACHE STRING "Executable that generates lldb-rpc-server")
  endif()
endif()

