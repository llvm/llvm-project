//===-- RPCServerHeaderEmitter.h ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_RPC_GEN_RPCSERVERHEADEREMITTER_H
#define LLDB_RPC_GEN_RPCSERVERHEADEREMITTER_H

#include "RPCCommon.h"

#include "clang/AST/AST.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace clang;

namespace lldb_rpc_gen {
/// Emit the source code for server-side *.h files.
class RPCServerHeaderEmitter : public FileEmitter {
public:
  RPCServerHeaderEmitter(std::unique_ptr<llvm::ToolOutputFile> &&OutputFile)
      : FileEmitter(std::move(OutputFile)) {
    Begin();
  }

  ~RPCServerHeaderEmitter() { End(); }

  void EmitMethod(const Method &method);

private:
  void EmitHandleRPCCall();

  void EmitConstructor(const std::string &MangledName);

  void EmitDestructor(const std::string &MangledName);

  std::string GetHeaderGuard();

  void Begin();

  void End();
};
} // namespace lldb_rpc_gen

#endif // LLDB_RPC_GEN_RPCSERVERHEADEREMITTER_H
