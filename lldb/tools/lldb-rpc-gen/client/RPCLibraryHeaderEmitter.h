#ifndef LLDB_RPC_GEN_RPCLIBRARYHEADEREMITTER_H
#define LLDB_RPC_GEN_RPCLIBRARYHEADEREMITTER_H

#include "RPCCommon.h"

#include "clang/AST/AST.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace clang;

namespace lldb_rpc_gen {

class RPCLibraryHeaderEmitter : public FileEmitter {
public:
  RPCLibraryHeaderEmitter(std::unique_ptr<llvm::ToolOutputFile> &&OutputFile)
      : FileEmitter(std::move(OutputFile)), CurrentClass() {
    Begin();
  }

  ~RPCLibraryHeaderEmitter() { End(); }

  void StartClass(std::string ClassName);

  void EndClass();

  void EmitMethod(const Method &method);

  void EmitEnum(EnumDecl *E);

private:
  std::string GetHeaderGuard();

  void Begin();

  void End();

  std::string CurrentClass;
  bool CopyCtorEmitted = false;
  bool CopyAssignEmitted = false;
  bool MoveCtorEmitted = false;
  bool MoveAssignEmitted = false;
};
} // namespace lldb_rpc_gen
#endif // LLDB_RPC_GEN_RPCLIBRARYHEADEREMITTER_H
