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
add_dependencies(lldb-rpc-gen clang-resource-headers)
setup_host_tool(lldb-rpc-gen LLDB_RPC_GEN_EXE lldb_rpc_gen_exe lldb_rpc_gen_target)
