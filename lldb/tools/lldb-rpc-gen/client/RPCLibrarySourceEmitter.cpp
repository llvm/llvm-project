#include "RPCLibrarySourceEmitter.h"
#include "RPCCommon.h"

#include "clang/AST/AST.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace clang;
using namespace lldb_rpc_gen;

static constexpr llvm::StringRef ReturnVariableName("__result");

// This map stores any method that needs custom logic with a struct that
// tells us where the logic needs to be inserted and what code needs to be
// inserted. The code here is stored as a raw string literal.
const llvm::StringMap<RPCLibrarySourceEmitter::CustomLogic>
    CustomLogicForMethods = {
        {"_ZN4lldb10SBDebugger6CreateEbPFvPKcPvES3_",
         {RPCLibrarySourceEmitter::CustomLogicLocation::eAfterDecode, R"code(
    // Now source the .lldbinit files manually since we can't rely on the
    // LLDB.framework on the other side to have special support for sourcing the right file
    // since it would try to source "~/.lldbinit-lldb-rpc-server" followed by
    // "~/.lldbinit". We want it to try "~.lldbinit-%s" where %s is the
    // current program basename followed by "~/.lldbinit".

    if (source_init_files && __result.ObjectRefIsValid()) {
      const char *program_basename = rpc::GetProgramBasename();
      if (program_basename) {
        char init_path[PATH_MAX];
        snprintf(init_path, sizeof(init_path), "~/.lldbinit-%s",
                 program_basename);
        lldb_rpc::SBFileSpec program_init_file(connection, init_path, true);
        if (program_init_file.Exists()) {
          char command_str[PATH_MAX];
          snprintf(command_str, sizeof(command_str),
                   "command source -s 1 -c 1 -e 0 '%s'", init_path);
          __result.HandleCommand(command_str);
        } else {
          __result.HandleCommand("command source -s 1 -c 1 -e 0 '~/.lldbinit'");
        }
      }
    })code"}},
};

static std::string GetLocalObjectRefCtor(const std::string &ClassName) {
  return "rpc::LocalObjectRef(LLDB_RPC_INVALID_CONNECTION_ID, eClass_lldb_" +
         ClassName + ", LLDB_RPC_INVALID_OBJECT_ID)";
}

static std::string GetObjectRefCtor(const std::string &ClassName) {
  return "rpc::ObjectRef(LLDB_RPC_INVALID_CONNECTION_ID, eClass_lldb_" +
         ClassName + ", LLDB_RPC_INVALID_OBJECT_ID)";
}

void RPCLibrarySourceEmitter::EmitCopyCtor() {
  EmitLine("lldb_rpc::" + CurrentClass + "::" + CurrentClass +
           "(const lldb_rpc::" + CurrentClass + " &rhs) = default;");
}

void RPCLibrarySourceEmitter::EmitCopyAssign() {
  EmitLine("lldb_rpc::" + CurrentClass + " &lldb_rpc::" + CurrentClass +
           "::operator=(const lldb_rpc::" + CurrentClass + " &rhs) = default;");
}

void RPCLibrarySourceEmitter::EmitMoveCtor() {
  EmitLine("lldb_rpc::" + CurrentClass + "::" + CurrentClass +
           "(lldb_rpc::" + CurrentClass + " &&rhs) = default;");
}

void RPCLibrarySourceEmitter::EmitMoveAssign() {
  EmitLine("lldb_rpc::" + CurrentClass + " &lldb_rpc::" + CurrentClass +
           "::operator=(lldb_rpc::" + CurrentClass + " &&rhs) = default;");
}

void RPCLibrarySourceEmitter::EmitMethod(const Method &method) {
  if (method.IsCopyCtor) {
    CopyCtorEmitted = true;
    EmitCopyCtor();
    return;
  } else if (method.IsCopyAssign) {
    CopyAssignEmitted = true;
    EmitCopyAssign();
    return;
  } else if (method.IsMoveCtor) {
    MoveCtorEmitted = true;
    EmitMoveCtor();
    return;
  } else if (method.IsMoveAssign) {
    MoveAssignEmitted = true;
    EmitMoveAssign();
    return;
  }

  EmitCommentHeader(method);
  EmitFunctionHeader(method);
  EmitFunctionBody(method);
  EmitFunctionFooter();
}

void RPCLibrarySourceEmitter::StartClass(std::string ClassName) {
  CurrentClass = std::move(ClassName);
  if (lldb_rpc_gen::SBClassRequiresDefaultCtor(CurrentClass)) {
    std::string BaseClassCtor =
        lldb_rpc_gen::SBClassInheritsFromObjectRef(CurrentClass)
            ? GetObjectRefCtor(CurrentClass)
            : GetLocalObjectRefCtor(CurrentClass);
    EmitLine("lldb_rpc::" + CurrentClass + "::" + CurrentClass +
             "() : " + BaseClassCtor + " {}");
  }
}

void RPCLibrarySourceEmitter::EndClass() {
  if (lldb_rpc_gen::SBClassRequiresCopyCtorAssign(CurrentClass)) {
    if (!CopyCtorEmitted)
      EmitCopyCtor();

    if (!CopyAssignEmitted)
      EmitCopyAssign();
  }

  if (!MoveCtorEmitted)
    EmitMoveCtor();

  if (!MoveAssignEmitted)
    EmitMoveAssign();

  CopyCtorEmitted = false;
  CopyAssignEmitted = false;
  MoveCtorEmitted = false;
  MoveAssignEmitted = false;
}

void RPCLibrarySourceEmitter::EmitCommentHeader(const Method &method) {
  std::string CommentLine;
  llvm::raw_string_ostream CommentStream(CommentLine);

  CommentStream << "// "
                << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
                       method.QualifiedName)
                << "(" << method.CreateParamListAsString(eLibrary) << ")";
  if (method.IsConst)
    CommentStream << " const";

  EmitLine("//-----------------------------------------------------------");
  EmitLine(CommentLine);
  EmitLine("//-----------------------------------------------------------");
}

void RPCLibrarySourceEmitter::EmitFunctionHeader(const Method &method) {
  std::string FunctionHeader;
  llvm::raw_string_ostream FunctionHeaderStream(FunctionHeader);

  if (!method.IsDtor && !method.IsConversionMethod && !method.IsCtor)
    FunctionHeaderStream << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
                                method.ReturnType.getAsString(method.Policy))
                         << " ";

  FunctionHeaderStream << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
                              method.QualifiedName)
                       << "(" << method.CreateParamListAsString(eLibrary)
                       << ")";
  if (method.IsConst)
    FunctionHeaderStream << " const";
  if (method.IsCtor) {
    if (lldb_rpc_gen::SBClassInheritsFromObjectRef(method.BaseName))
      FunctionHeaderStream << " : " << GetObjectRefCtor(method.BaseName);
    else
      FunctionHeaderStream << " : " << GetLocalObjectRefCtor(method.BaseName);
  }
  FunctionHeaderStream << " {";

  EmitLine(FunctionHeader);
  IndentLevel++;
}

void RPCLibrarySourceEmitter::EmitFunctionBody(const Method &method) {
  // There's nothing to do for destructors. The LocalObjectRef destructor should
  // handle everything for us.
  if (method.IsDtor)
    return;

  EmitLine("// 1) Perform setup");
  EmitFunctionSetup(method);
  EmitLine("// 2) Send RPC call");
  EmitSendRPCCall(method);
  EmitLine("// 3) Decode return values");
  EmitDecodeReturnValues(method);
}

void RPCLibrarySourceEmitter::EmitFunctionFooter() {
  IndentLevel--;
  EmitLine("}");
}

void RPCLibrarySourceEmitter::EmitFunctionSetup(const Method &method) {
  if (!method.ReturnType->isVoidType())
    EmitReturnValueStorage(method);

  EmitConnectionSetup(method);

  EmitLine("// RPC Communication setup");
  EmitLine("static RPCFunctionInfo g_func(\"" + method.MangledName + "\");");
  EmitLine("RPCStream send;");
  EmitLine("RPCStream response;");
  EmitLine("g_func.Encode(send);");

  EmitEncodeParameters(method);

  if (CustomLogicForMethods.lookup(method.MangledName).Location ==
      CustomLogicLocation::eAfterSetup)
    EmitCustomLogic(method);
}

void RPCLibrarySourceEmitter::EmitReturnValueStorage(const Method &method) {
  assert(!method.ReturnType->isVoidType() &&
         "Cannot emit return value storage when return type is 'void'");

  EmitLine("// Storage for return value");
  std::string ReturnValueStorage;
  llvm::raw_string_ostream ReturnValueStorageStream(ReturnValueStorage);

  std::string ReturnValueType;
  if (lldb_rpc_gen::TypeIsConstCharPtr(method.ReturnType))
    ReturnValueStorageStream << "rpc_common::ConstCharPointer "
                             << ReturnVariableName << ";";
  else if (method.ReturnType->isPointerType())
    ReturnValueStorageStream << "Bytes " << ReturnVariableName << ";";
  else {
    // We need to get the unqualified type because we don't want the return
    // variable to be marked `const`. That would prevent us from changing it
    // during the decoding step.
    QualType UnqualifiedReturnType = method.ReturnType.getUnqualifiedType();
    ReturnValueStorageStream
        << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
               UnqualifiedReturnType.getAsString(method.Policy))
        << " " << ReturnVariableName << " = {};";
  }
  EmitLine(ReturnValueStorage);
}

void RPCLibrarySourceEmitter::EmitConnectionSetup(const Method &method) {
  // Methods know if they require a connection parameter. We need to figure out
  // which scenario we're in.
  bool ConnectionDerived = false;
  if (!method.RequiresConnectionParameter()) {
    // If we have an instance method that is not a constructor, we have a valid
    // connection from `this` via `ObjectRefGetConnectionSP()`.
    if (!method.IsCtor && method.IsInstance) {
      EmitLine("// Deriving connection from this.");
      EmitLine("rpc_common::ConnectionSP connection_sp = "
               "ObjectRefGetConnectionSP();");
      ConnectionDerived = true;
    }

    // Otherewise, we try to derive it from an existing parameter.
    if (!ConnectionDerived)
      for (const auto &Param : method.Params) {
        if (lldb_rpc_gen::TypeIsSBClass(Param.Type)) {
          EmitLine("// Deriving connection from SB class parameter.");
          std::string ConnectionLine =
              "rpc_common::ConnectionSP connection_sp = " + Param.Name;
          if (Param.Type->isPointerType())
            ConnectionLine += "->";
          else
            ConnectionLine += ".";
          ConnectionLine += "ObjectRefGetConnectionSP();";
          EmitLine(ConnectionLine);
          ConnectionDerived = true;
          break;
        }
      }
  } else {
    // This method requires a connection parameter. It will always be named
    // "connection" and it will always come first in the parameter list.
    EmitLine("// Using connection parameter.");
    EmitLine(
        "rpc_common::ConnectionSP connection_sp = connection.GetConnection();");
    ConnectionDerived = true;
  }

  assert(ConnectionDerived &&
         "Unable to determine where method should derive connection from");

  // NOTE: By this point, we should have already emitted the storage for the
  // return value.
  std::string FailureReturnExpression;
  if (method.ReturnType->isPointerType())
    FailureReturnExpression = "nullptr";
  else if (!method.ReturnType->isVoidType())
    FailureReturnExpression = "__result";
  EmitLine("if (!connection_sp) return " + FailureReturnExpression + ";");
}

void RPCLibrarySourceEmitter::EmitEncodeParameters(const Method &method) {
  // Encode parameters.
  if (method.IsInstance && !method.IsCtor)
    EmitLine("RPCValueEncoder(send, "
             "rpc_common::RPCPacket::ValueType::Argument, *this);");

  for (auto Iter = method.Params.begin(); Iter != method.Params.end(); Iter++) {
    // SBTarget::BreakpointCreateByNames specifically uses an
    // rpc_common::StringList when encoding the list of symbol names in the
    // handwritten version. This allows it to account for null terminators and
    // without it, Xcode crashes when calling this function. The else-if block
    // below replaces params that have a pointer and length with a Bytes object.
    // We can use the same logic in order to replace const char **s with
    // StringLists
    if (lldb_rpc_gen::TypeIsConstCharPtrPtr(Iter->Type) &&
        Iter->IsFollowedByLen) {
      std::string StringListLine;
      const std::string StringListName = Iter->Name + "_list";
      StringListLine = "StringList " + StringListName + "(" + Iter->Name + ", ";
      Iter++;
      StringListLine += Iter->Name + ");";
      Iter--;
      EmitLine(StringListLine);
      EmitLine(
          "RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, " +
          StringListName + ");");
      // When we have pointer parameters, in general the strategy is
      // to create `Bytes` objects from them (basically a buffer with a size)
      // and then move those over the wire. We're not moving the pointer itself,
      // but the contents of memory being pointed to. There are a few exceptions
      // to this:
      //   - If the type is `const char *` or `const char **`, those are handled
      //     specially and can be encoded directly.
      //   - If we have a function pointer, we move the pointer value directly.
      //     To do the callback from the server-side, we will need this pointer
      //     value to correctly invoke the function client-side.
      //   - If we have a baton (to support a callback), we need to move the
      //     pointer value directly. This is for the same reason as callbacks
      //     above.
      //   - If we have a pointer to an SB class, we just dereference it and
      //     encode it like a normal SB class object.
    } else if (Iter->Type->isPointerType() &&
               !Iter->Type->isFunctionPointerType() &&
               (!Iter->Type->isVoidPointerType() || Iter->IsFollowedByLen) &&
               !lldb_rpc_gen::TypeIsConstCharPtr(Iter->Type) &&
               !lldb_rpc_gen::TypeIsConstCharPtrPtr(Iter->Type) &&
               !lldb_rpc_gen::TypeIsSBClass(Iter->Type)) {
      // NOTE: We're aiming for this transformation:
      //   (TYPE *buf, SIZE len) -> Bytes(buf, sizeof(TYPE) * len)
      // The `sizeof` portion is dropped when TYPE is `void`. When there is no
      // length argument, we implicitly assume that `len` is 1.
      const std::string BufferName = Iter->Name + "_buffer";
      QualType UnderlyingType = lldb_rpc_gen::GetUnderlyingType(Iter->Type);
      std::string BufferLine = "Bytes " + BufferName + "(" + Iter->Name + ", ";
      if (!Iter->Type->isVoidPointerType())
        BufferLine += "sizeof(" +
                      ReplaceLLDBNamespaceWithRPCNamespace(
                          UnderlyingType.getAsString(method.Policy)) +
                      ")";

      if (Iter->IsFollowedByLen && !Iter->Type->isVoidPointerType())
        BufferLine += " * ";

      if (Iter->IsFollowedByLen) {
        Iter++;
        BufferLine += Iter->Name;
      }

      BufferLine += ");";
      EmitLine(BufferLine);
      EmitLine("RPCValueEncoder(send, "
               "rpc_common::RPCPacket::ValueType::Argument, " +
               BufferName + ");");
    } else if (lldb_rpc_gen::TypeIsSBClass(Iter->Type) &&
               Iter->Type->isPointerType()) {
      // If we have a pointer to an SB class, the strategy is to check for
      // nullptr. If we have a valid pointer, we just encode the actual SB class
      // itself. Otherwise, we'll need to create a blank one and send that
      // along.
      // Note: Currently all methods that take SB class pointers are instance
      // methods. This assertion will fail if that changes.
      assert(method.IsInstance &&
             "Assumption that only instance methods have pointers to SB class "
             "as parameters is no longer true. Please update this logic.");
      EmitLine("if (" + Iter->Name + ")");
      IndentLevel++;
      EmitLine("RPCValueEncoder(send, "
               "rpc_common::RPCPacket::ValueType::Argument, *" +
               Iter->Name + ");");
      IndentLevel--;
      EmitLine("else");
      IndentLevel++;
      // FIXME: change this logic, not everything can be an rpc::ObjectRef!!!
      const std::string ClassIdentifier =
          "eClass_lldb_" + lldb_rpc_gen::GetSBClassNameFromType(Iter->Type);
      EmitLine(
          "RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, "
          "rpc::ObjectRef(ObjectRefGetConnectionID(), " +
          ClassIdentifier + ", LLDB_RPC_INVALID_OBJECT_ID));");
      IndentLevel--;
    } else {
      const std::string CallbackCast = Iter->Type->isFunctionPointerType()
                                           ? "(rpc_common::function_ptr_t)"
                                           : "";

      // NOTE: We're going to assume that SB objects are coming in with a valid
      // connection and we'll assert if they don't have one.
      if (lldb_rpc_gen::TypeIsSBClass(Iter->Type)) {
        std::string TypeName = lldb_rpc_gen::GetSBClassNameFromType(Iter->Type);
        EmitLine("assert(" + Iter->Name +
                 ".ObjectRefIsValid() && \"SB object refs must be valid before "
                 "encoding\");");
      }

      EmitLine(
          "RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, " +
          CallbackCast + Iter->Name + ");");
    }
  }
}

void RPCLibrarySourceEmitter::EmitSendRPCCall(const Method &method) {
  EmitLine(
      "if (!connection_sp->SendRPCCallAndWaitForResponse(send, response))");
  IndentLevel++;
  if (method.ReturnType->isVoidType() || method.IsCtor)
    EmitLine("return;");
  else if (method.ReturnType->isPointerType())
    EmitLine("return nullptr;");
  else
    EmitLine("return __result;");
  IndentLevel--;
  if (CustomLogicForMethods.lookup(method.MangledName).Location ==
      CustomLogicLocation::eAfterRPCCall)
    EmitCustomLogic(method);
}

void RPCLibrarySourceEmitter::EmitDecodeReturnValues(const Method &method) {
  if (!method.ReturnType->isVoidType()) {
    std::string DecodeReturnValueLine =
        "RPCValueDecoder(response, "
        "rpc_common::RPCPacket::ValueType::ReturnValue, " +
        ReturnVariableName.str() + ");";
    EmitLine(DecodeReturnValueLine);
  } else if (method.IsCtor)
    EmitLine("RPCValueDecoder(response, "
             "rpc_common::RPCPacket::ValueType::ReturnValue, *this);");

  // Update mutable parameters (references and pointers)
  for (auto Iter = method.Params.begin(); Iter != method.Params.end(); Iter++) {
    // If what we have is not a reference type or a pointer type, we can safely
    // skip this.
    if (!Iter->Type->isReferenceType() && !Iter->Type->isPointerType())
      continue;

    // No need to update SB class instances on the client-side.
    if (lldb_rpc_gen::TypeIsSBClass(Iter->Type))
      continue;

    // We skip over function pointers and their accompanying baton parameters.
    if (Iter->Type->isFunctionPointerType() ||
        (Iter->Type->isVoidPointerType() && !Iter->IsFollowedByLen))
      continue;

    // We cannot update const-qualified parameters.
    QualType UnderlyingType = lldb_rpc_gen::GetUnderlyingType(Iter->Type);
    // This is specific to pointers, but we need to get to the innermost type to
    // get the qualification. For references, this loop will never execute.
    while (UnderlyingType->isPointerType())
      UnderlyingType = lldb_rpc_gen::GetUnderlyingType(UnderlyingType);

    if (UnderlyingType.isConstQualified())
      continue;

    if (Iter->Type->isReferenceType())
      EmitLine("RPCValueDecoder(response, "
               "rpc_common::RPCPacket::ValueType::ReturnValue, " +
               Iter->Name + ");");
    else {
      assert(Iter->Type->isPointerType() &&
             "Mutable parameter is not reference or pointer!");
      const std::string &PointerParameterName = Iter->Name;
      const std::string BufferName = PointerParameterName + "_buffer";
      std::string SizeExpression;

      // If we have a `void *` with a length parameter, we are counting bytes.
      // No `sizeof` will be needed.
      if (!Iter->Type->isVoidPointerType()) {
        QualType UnderlyingType = lldb_rpc_gen::GetUnderlyingType(Iter->Type);
        SizeExpression = "sizeof(" +
                         ReplaceLLDBNamespaceWithRPCNamespace(
                             UnderlyingType.getAsString(method.Policy)) +
                         ")";
      }

      if (Iter->IsFollowedByLen) {
        // If we have a sizeof, we must multiply by the length argument.
        if (!SizeExpression.empty())
          SizeExpression += " * ";
        Iter++;
        SizeExpression += Iter->Name;
      }

      EmitLine("RPCValueDecoder(response, "
               "rpc_common::RPCPacket::ValueType::ReturnValue, " +
               BufferName + ");");
      EmitLine("assert(" + BufferName + ".GetSize() == " + SizeExpression +
               " && \"Buffer was resized during RPC call\");");
      // NOTE: We can just treat the pointers as `void *` and copy all the bytes
      // needed.
      EmitLine("memcpy(" + PointerParameterName + ", " + BufferName +
               ".GetData(), " + BufferName + ".GetSize());");
    }
  }

  if (CustomLogicForMethods.lookup(method.MangledName).Location ==
      CustomLogicLocation::eAfterDecode)
    EmitCustomLogic(method);
  if (!method.ReturnType->isVoidType()) {
    // FIXME: Find a solution that does not involve leaking memory.
    // We have to persist the buffer we returned somewhere. We stick it in the
    // RPCStringPool for now
    if (method.ReturnType->isPointerType()) {
      std::string ReturnExpression =
          "return (" +
          lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
              method.ReturnType.getAsString(method.Policy)) +
          ")RPCStringPool::Add(" + ReturnVariableName.str() + ");";
      EmitLine(ReturnExpression);
    } else
      EmitLine("return __result;");
  }
}

void RPCLibrarySourceEmitter::EmitCustomLogic(const Method &method) {
  assert(CustomLogicForMethods.contains(method.MangledName) &&
         "Cannot emit custom logic for method that is not present in custom "
         "logic map.");
  EmitLine("// Custom logic:");
  EmitLine(CustomLogicForMethods.lookup(method.MangledName).Code);
}
