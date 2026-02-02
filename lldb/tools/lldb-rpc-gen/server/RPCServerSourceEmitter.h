//===-- RPCServerSourceEmitter.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
#ifndef LLDB_RPC_GEN_RPCSERVERMETHODEMITTER_H
#define LLDB_RPC_GEN_RPCSERVERMETHODEMITTER_H

#include "RPCCommon.h"

#include "clang/AST/AST.h"

#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace lldb_rpc_gen {
/// Emit the source code for server-side *.cpp files.
class RPCServerSourceEmitter : public FileEmitter {
public:
  RPCServerSourceEmitter(std::unique_ptr<llvm::ToolOutputFile> &&OutputFile)
      : FileEmitter(std::move(OutputFile)) {
    Begin();
  }

  /// Given a Method, emits a server-side implementation of the method
  /// for lldb-rpc-server
  void EmitMethod(const Method &method);

private:
  void EmitCommentHeader(const Method &method);

  void EmitFunctionHeader(const Method &method);

  void EmitFunctionBody(const Method &method);

  void EmitFunctionFooter();

  void EmitStorageForParameters(const Method &method);

  void EmitStorageForOneParameter(QualType ParamType,
                                  const std::string &ParamName,
                                  const PrintingPolicy &Policy,
                                  bool IsFollowedByLen);

  void EmitDecodeForParameters(const Method &method);

  void EmitDecodeForOneParameter(QualType ParamType,
                                 const std::string &ParamName,
                                 const PrintingPolicy &Policy);

  std::string CreateMethodCall(const Method &method);

  std::string CreateEncodeLine(const std::string &value,
                               bool IsEncodingSBClass);

  void EmitEncodesForMutableParameters(const std::vector<Param> &Params);

  void EmitMethodCallAndEncode(const Method &method);

  void EmitCallbackFunction(const Method &method);

  void Begin() {
    EmitLine("#include \"RPCUserServer.h\"");
    EmitLine("#include \"SBAPI.h\"");
    EmitLine("#include <lldb-rpc/common/RPCArgument.h>");
    EmitLine("#include <lldb-rpc/common/RPCCommon.h>");
    EmitLine("#include <lldb-rpc/common/RPCFunction.h>");
    EmitLine("#include <lldb/API/LLDB.h>");
    EmitLine("");
    EmitLine("using namespace rpc_common;");
    EmitLine("using namespace lldb;");
  }
};
} // namespace lldb_rpc_gen

#endif // LLDB_RPC_GEN_RPCSERVERMETHODEMITTER_H
