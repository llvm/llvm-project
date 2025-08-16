#ifndef LLDB_RPC_GEN_RPCLIBRARYSOURCEEMITTER_H
#define LLDB_RPC_GEN_RPCLIBRARYSOURCEEMITTER_H

#include "RPCCommon.h"

#include "clang/AST/AST.h"

#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;

namespace lldb_rpc_gen {
class RPCLibrarySourceEmitter : public FileEmitter {
public:
  RPCLibrarySourceEmitter(std::unique_ptr<llvm::ToolOutputFile> &&OutputFile)
      : FileEmitter(std::move(OutputFile)) {
    Begin();
  }

  enum class CustomLogicLocation {
    eNone,
    eAfterSetup,
    eAfterRPCCall,
    eAfterDecode
  };

  struct CustomLogic {
    CustomLogicLocation Location;
    std::string Code;
  };

  void StartClass(std::string ClassName);

  void EndClass();

  void EmitMethod(const Method &method);

  void EmitEmptyConstructor(const std::string &ClassName);

private:
  void EmitCopyCtor();

  void EmitCopyAssign();

  void EmitMoveCtor();

  void EmitMoveAssign();

  void EmitCommentHeader(const Method &method);

  void EmitFunctionHeader(const Method &method);

  void EmitFunctionBody(const Method &method);

  void EmitFunctionFooter();

  void EmitFunctionSetup(const Method &method);

  void EmitReturnValueStorage(const Method &method);

  void EmitConnectionSetup(const Method &method);

  void EmitEncodeParameters(const Method &method);

  void EmitSendRPCCall(const Method &method);

  void EmitDecodeReturnValues(const Method &method);

  void EmitCustomLogic(const Method &method);

  void Begin() {
    EmitLine("#include <lldb-rpc/common/RPCArgument.h>");
    EmitLine("#include <lldb-rpc/common/RPCCommon.h>");
    EmitLine("#include <lldb-rpc/common/RPCFunction.h>");
    EmitLine("#include <lldb-rpc/common/RPCStringPool.h>");
    EmitLine("#include <lldb-rpc/liblldbrpc/RPCUserClient.h>");
    EmitLine("#include \"LLDBRPC.h\"");
    EmitLine("#include <cassert>");
    EmitLine("using namespace rpc_common;");
    EmitLine("using namespace lldb_rpc;");
  }

  std::string CurrentClass;
  bool CopyCtorEmitted = false;
  bool CopyAssignEmitted = false;
  bool MoveCtorEmitted = false;
  bool MoveAssignEmitted = false;
};
} // namespace lldb_rpc_gen

#endif // LLDB_RPC_GEN_RPCLIBRARYSOURCEEMITTER_H
