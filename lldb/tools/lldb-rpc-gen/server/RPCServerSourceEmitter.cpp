//===-- RPCServerSourceEmitter.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPCServerSourceEmitter.h"
#include "RPCCommon.h"

#include "clang/AST/AST.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <map>

using namespace clang;
using namespace lldb_rpc_gen;

// For methods with pointer return types, it's important that we know how big
// the type of the pointee is. We must correctly size a buffer (in the form of a
// Bytes object) before we can actually use it.
static const std::map<llvm::StringRef, size_t> MethodsWithPointerReturnTypes = {
    {"_ZN4lldb12SBModuleSpec12GetUUIDBytesEv", 16}, // sizeof(uuid_t) -> 16
    {"_ZNK4lldb8SBModule12GetUUIDBytesEv", 16},     // sizeof(uuid_t) -> 16
};

void RPCServerSourceEmitter::EmitMethod(const Method &method) {
  if (method.ContainsFunctionPointerParameter)
    EmitCallbackFunction(method);

  EmitCommentHeader(method);
  EmitFunctionHeader(method);
  EmitFunctionBody(method);
  EmitFunctionFooter();
}

void RPCServerSourceEmitter::EmitCommentHeader(const Method &method) {
  std::string CommentLine;
  llvm::raw_string_ostream CommentStream(CommentLine);

  CommentStream << "// " << method.QualifiedName << "("
                << method.CreateParamListAsString(eServer) << ")";
  if (method.IsConst)
    CommentStream << " const";

  EmitLine("//------------------------------------------------------------");
  EmitLine(CommentLine);
  EmitLine("//------------------------------------------------------------");
}

void RPCServerSourceEmitter::EmitFunctionHeader(const Method &method) {
  std::string FunctionHeader;
  llvm::raw_string_ostream FunctionHeaderStream(FunctionHeader);
  FunctionHeaderStream
      << "bool rpc_server::" << method.MangledName
      << "::HandleRPCCall(rpc_common::Connection &connection, RPCStream "
         "&send, RPCStream &response) {";
  EmitLine(FunctionHeader);
  IndentLevel++;
}

void RPCServerSourceEmitter::EmitFunctionBody(const Method &method) {
  EmitLine("// 1) Make local storage for incoming function arguments");
  EmitStorageForParameters(method);
  EmitLine("// 2) Decode all function arguments");
  EmitDecodeForParameters(method);
  EmitLine("// 3) Call the method and encode the return value");
  EmitMethodCallAndEncode(method);
}

void RPCServerSourceEmitter::EmitFunctionFooter() {
  EmitLine("return true;");
  IndentLevel--;
  EmitLine("}");
}

void RPCServerSourceEmitter::EmitStorageForParameters(const Method &method) {
  // If we have an instance method and it isn't a constructor, we'll need to
  // emit a "this" pointer.
  if (method.IsInstance && !method.IsCtor)
    EmitStorageForOneParameter(method.ThisType, "this_ptr", method.Policy,
                               /* IsFollowedByLen = */ false);
  for (auto Iter = method.Params.begin(); Iter != method.Params.end(); Iter++) {
    EmitStorageForOneParameter(Iter->Type, Iter->Name, method.Policy,
                               Iter->IsFollowedByLen);
    // Skip over the length parameter, we don't emit it.
    if (!lldb_rpc_gen::TypeIsConstCharPtrPtr(Iter->Type) &&
        Iter->IsFollowedByLen)
      Iter++;
  }
}

void RPCServerSourceEmitter::EmitStorageForOneParameter(
    QualType ParamType, const std::string &ParamName,
    const PrintingPolicy &Policy, bool IsFollowedByLen) {
  // First, we consider `const char *`, `const char **`. They have special
  // server-side types.
  if (TypeIsConstCharPtr(ParamType)) {
    EmitLine("rpc_common::ConstCharPointer " + ParamName + ";");
    return;
  } else if (TypeIsConstCharPtrPtr(ParamType)) {
    EmitLine("rpc_common::StringList " + ParamName + ";");
    return;
  }

  QualType UnderlyingType =
      lldb_rpc_gen::GetUnqualifiedUnderlyingType(ParamType);
  const bool IsSBClass = lldb_rpc_gen::TypeIsSBClass(UnderlyingType);

  if (ParamType->isPointerType() && !IsSBClass) {
    // Void pointer with no length is usually a baton for a callback. We're
    // going to hold onto the pointer value so we can send it back to the
    // client-side when we implement callbacks.
    if (ParamType->isVoidPointerType() && !IsFollowedByLen) {
      EmitLine("void * " + ParamName + " = nullptr;");
      return;
    }

    if (!ParamType->isFunctionPointerType()) {
      EmitLine("Bytes " + ParamName + ";");
      return;
    }

    assert(ParamType->isFunctionPointerType() && "Unhandled pointer type");
    EmitLine("rpc_common::function_ptr_t " + ParamName + " = nullptr;");
    return;
  }

  std::string StorageDeclaration;
  llvm::raw_string_ostream StorageDeclarationStream(StorageDeclaration);

  UnderlyingType.print(StorageDeclarationStream, Policy);
  StorageDeclarationStream << " ";
  if (IsSBClass)
    StorageDeclarationStream << "*";
  StorageDeclarationStream << ParamName;
  if (IsSBClass)
    StorageDeclarationStream << " = nullptr";
  else
    StorageDeclarationStream << " = {}";
  StorageDeclarationStream << ";";
  EmitLine(StorageDeclaration);
}

void RPCServerSourceEmitter::EmitDecodeForParameters(const Method &method) {
  if (method.IsInstance && !method.IsCtor)
    EmitDecodeForOneParameter(method.ThisType, "this_ptr", method.Policy);
  for (auto Iter = method.Params.begin(); Iter != method.Params.end(); Iter++) {
    EmitDecodeForOneParameter(Iter->Type, Iter->Name, method.Policy);
    if (!lldb_rpc_gen::TypeIsConstCharPtrPtr(Iter->Type) &&
        Iter->IsFollowedByLen)
      Iter++;
  }
}

void RPCServerSourceEmitter::EmitDecodeForOneParameter(
    QualType ParamType, const std::string &ParamName,
    const PrintingPolicy &Policy) {
  QualType UnderlyingType =
      lldb_rpc_gen::GetUnqualifiedUnderlyingType(ParamType);

  if (TypeIsSBClass(UnderlyingType)) {
    std::string DecodeLine;
    llvm::raw_string_ostream DecodeLineStream(DecodeLine);
    DecodeLineStream << ParamName << " = "
                     << "RPCServerObjectDecoder<";
    UnderlyingType.print(DecodeLineStream, Policy);
    DecodeLineStream << ">(send, rpc_common::RPCPacket::ValueType::Argument);";
    EmitLine(DecodeLine);
    EmitLine("if (!" + ParamName + ")");
    IndentLevel++;
    EmitLine("return false;");
    IndentLevel--;
  } else {
    EmitLine("if (!RPCValueDecoder(send, "
             "rpc_common::RPCPacket::ValueType::Argument, " +
             ParamName + "))");
    IndentLevel++;
    EmitLine("return false;");
    IndentLevel--;
  }
}

std::string RPCServerSourceEmitter::CreateMethodCall(const Method &method) {
  std::string MethodCall;
  llvm::raw_string_ostream MethodCallStream(MethodCall);
  if (method.IsInstance) {
    if (!method.IsCtor)
      MethodCallStream << "this_ptr->";
    MethodCallStream << method.BaseName;
  } else
    MethodCallStream << method.QualifiedName;

  std::vector<std::string> Args;
  std::string FunctionPointerName;
  for (auto Iter = method.Params.begin(); Iter != method.Params.end(); Iter++) {
    std::string Arg;
    // We must check for `const char *` and `const char **` first.
    if (TypeIsConstCharPtr(Iter->Type)) {
      // `const char *` is stored server-side as rpc_common::ConstCharPointer
      Arg = Iter->Name + ".c_str()";
    } else if (TypeIsConstCharPtrPtr(Iter->Type)) {
      // `const char **` is stored server-side as rpc_common::StringList
      Arg = Iter->Name + ".argv()";
    } else if (lldb_rpc_gen::TypeIsSBClass(Iter->Type)) {
      Arg = Iter->Name;
      if (!Iter->Type->isPointerType())
        Arg = "*" + Iter->Name;
    } else if (Iter->Type->isPointerType() &&
               !Iter->Type->isFunctionPointerType() &&
               (!Iter->Type->isVoidPointerType() || Iter->IsFollowedByLen)) {
      // We move pointers between the server and client as 'Bytes' objects.
      // Pointers with length arguments will have their length filled in below.
      // Pointers with no length arguments are assumed to behave like an array
      // with length of 1, except for void pointers which are handled
      // differently.
      Arg = "(" + Iter->Type.getAsString(method.Policy) + ")" + Iter->Name +
            ".GetData()";
    } else if (Iter->Type->isFunctionPointerType()) {
      // If we have a function pointer, we only want to pass something along if
      // we got a real pointer.
      Arg = Iter->Name + " ? " + method.MangledName + "_callback : nullptr";
      FunctionPointerName = Iter->Name;
    } else if (Iter->Type->isVoidPointerType() && !Iter->IsFollowedByLen &&
               method.ContainsFunctionPointerParameter) {
      // Assumptions:
      //  - This is assumed to be the baton for the function pointer.
      //  - This is assumed to come after the function pointer parameter.
      // We always produce this regardless of the value of the baton argument.
      Arg = "new CallbackInfo(" + FunctionPointerName + ", " + Iter->Name +
            ", connection.GetConnectionID())";
    } else
      Arg = Iter->Name;

    if (Iter->Type->isRValueReferenceType())
      Arg = "std::move(" + Arg + ")";
    Args.push_back(Arg);

    if (!lldb_rpc_gen::TypeIsConstCharPtrPtr(Iter->Type) &&
        Iter->IsFollowedByLen) {
      std::string LengthArg = Iter->Name + ".GetSize()";
      if (!Iter->Type->isVoidPointerType()) {
        QualType UUT = lldb_rpc_gen::GetUnqualifiedUnderlyingType(Iter->Type);
        LengthArg += " / sizeof(" + UUT.getAsString(method.Policy) + ")";
      }
      Args.push_back(LengthArg);
      Iter++;
    }
  }
  MethodCallStream << "(" << llvm::join(Args, ", ") << ")";

  return MethodCall;
}

std::string RPCServerSourceEmitter::CreateEncodeLine(const std::string &Value,
                                                     bool IsEncodingSBClass) {
  std::string EncodeLine;
  llvm::raw_string_ostream EncodeLineStream(EncodeLine);

  if (IsEncodingSBClass)
    EncodeLineStream << "RPCServerObjectEncoder(";
  else
    EncodeLineStream << "RPCValueEncoder(";

  EncodeLineStream
      << "response, rpc_common::RPCPacket::ValueType::ReturnValue, ";
  EncodeLineStream << Value;
  EncodeLineStream << ");";
  return EncodeLine;
}

// There are 4 cases to consider:
// - const SBClass &: No need to do anything.
// - const foo &: No need to do anything.
// - SBClass &: The server and the client hold on to IDs to refer to specific
//   instances, so there's no need to send any information back to the client.
// - foo &: The client is sending us a value over the wire, but because the type
//   is mutable, we must send the changed value back in case the method call
//   mutated it.
//
// Updating a mutable reference is done as a return value from the RPC
// perspective. These return values need to be emitted after the method's return
// value, and they are emitted in the order in which they occur in the
// declaration.
void RPCServerSourceEmitter::EmitEncodesForMutableParameters(
    const std::vector<Param> &Params) {
  for (auto Iter = Params.begin(); Iter != Params.end(); Iter++) {
    // No need to manually update an SBClass
    if (lldb_rpc_gen::TypeIsSBClass(Iter->Type))
      continue;

    if (!Iter->Type->isReferenceType() && !Iter->Type->isPointerType())
      continue;

    // If we have a void pointer with no length, there's nothing to update. This
    // is likely a baton for a callback. The same goes for function pointers.
    if (Iter->Type->isFunctionPointerType() ||
        (Iter->Type->isVoidPointerType() && !Iter->IsFollowedByLen))
      continue;

    // No need to update pointers and references to const-qualified data.
    QualType UnderlyingType = lldb_rpc_gen::GetUnderlyingType(Iter->Type);
    if (UnderlyingType.isConstQualified())
      continue;

    const std::string EncodeLine =
        CreateEncodeLine(Iter->Name, /* IsEncodingSBClass = */ false);
    EmitLine(EncodeLine);
  }
}

// There are 3 possible scenarios that this method can encounter:
// 1. The method has no return value and is not a constructor.
//    Only the method call itself is emitted.
// 2. The method is a constructor.
//    The call to the constructor is emitted in the encode line.
// 3. The method has a return value.
//    The method call is emitted and the return value is captured in a variable.
//    After that, an encode call is emitted with the variable that captured the
//    return value.
void RPCServerSourceEmitter::EmitMethodCallAndEncode(const Method &method) {
  const std::string MethodCall = CreateMethodCall(method);

  // If this function returns nothing, we just emit the call and update any
  // mutable references. Note that constructors have return type `void` so we
  // must explicitly check for that here.
  if (!method.IsCtor && method.ReturnType->isVoidType()) {
    EmitLine(MethodCall + ";");
    EmitEncodesForMutableParameters(method.Params);
    return;
  }

  static constexpr llvm::StringLiteral ReturnVariableName("__result");

  // If this isn't a constructor, we'll need to store the result of the method
  // call in a result variable.
  if (!method.IsCtor) {
    // We need to determine what the appropriate return type is. Here is the
    // strategy:
    // 1.) `SBFoo` -> `SBFoo &&`
    // 2.) If the type is a pointer other than `const char *` or `const char **`
    //     or `void *`, the return type will be `Bytes` (e.g. `const uint8_t *`
    //     -> `Bytes`).
    // 3.) Otherwise, emit the exact same return type.
    std::string ReturnTypeName;
    std::string AssignLine;
    llvm::raw_string_ostream AssignLineStream(AssignLine);
    if (method.ReturnType->isPointerType() &&
        !lldb_rpc_gen::TypeIsConstCharPtr(method.ReturnType) &&
        !lldb_rpc_gen::TypeIsConstCharPtrPtr(method.ReturnType) &&
        !method.ReturnType->isVoidPointerType()) {
      llvm::StringRef MangledNameRef(method.MangledName);
      auto Pos = MethodsWithPointerReturnTypes.find(MangledNameRef);
      assert(Pos != MethodsWithPointerReturnTypes.end() &&
             "Unable to determine the size of the return buffer");
      if (Pos == MethodsWithPointerReturnTypes.end()) {
        EmitLine(
            "// Intentionally inserting a compiler error. lldb-rpc-gen "
            "was unable to determine how large the return buffer should be.");
        EmitLine("#error: \"unable to determine size of return buffer\"");
        return;
      }
      AssignLineStream << "Bytes " << ReturnVariableName << "(" << MethodCall
                       << ", " << Pos->second << ");";
    } else {
      if (lldb_rpc_gen::TypeIsSBClass(method.ReturnType)) {
        // We want to preserve constness, so we don't strip qualifications from
        // the underlying type
        QualType UnderlyingReturnType =
            lldb_rpc_gen::GetUnderlyingType(method.ReturnType);
        ReturnTypeName =
            UnderlyingReturnType.getAsString(method.Policy) + " &&";
      } else
        ReturnTypeName = method.ReturnType.getAsString(method.Policy);

      AssignLineStream << ReturnTypeName << " " << ReturnVariableName << " = "
                       << MethodCall << ";";
    }
    EmitLine(AssignLine);
  }

  const bool IsEncodingSBClass =
      lldb_rpc_gen::TypeIsSBClass(method.ReturnType) || method.IsCtor;

  std::string ValueToEncode;
  if (IsEncodingSBClass) {
    if (method.IsCtor)
      ValueToEncode = MethodCall;
    else
      ValueToEncode = "std::move(" + ReturnVariableName.str() + ")";
  } else
    ValueToEncode = ReturnVariableName.str();

  const std::string ReturnValueEncodeLine =
      CreateEncodeLine(ValueToEncode, IsEncodingSBClass);
  EmitLine(ReturnValueEncodeLine);
  EmitEncodesForMutableParameters(method.Params);
}

// NOTE: This contains most of the same knowledge as RPCLibrarySourceEmitter. I
// have chosen not to re-use code here because the needs are different enough
// that it would be more work to re-use than just reimplement portions of it.
// Specifically:
//  - Callbacks do not neatly fit into a `Method` object, which currently
//    assumes that you have a CXXMethodDecl (We have a FunctionDecl at most).
//  - We only generate callbacks that have a `void *` baton parameter. We hijack
//    those baton parameters and treat them differently.
//  - Callbacks need to do something special for moving SB class references back
//    to the client-side.
void RPCServerSourceEmitter::EmitCallbackFunction(const Method &method) {
  // Check invariants and locate necessary resources
  Param FuncPointerParam;
  Param BatonParam;
  for (const auto &Param : method.Params)
    if (Param.Type->isFunctionPointerType())
      FuncPointerParam = Param;
    else if (Param.Type->isVoidPointerType())
      BatonParam = Param;

  assert(FuncPointerParam.Type->isFunctionPointerType() &&
         "Emitting callback function with no function pointer");
  assert(BatonParam.Type->isVoidPointerType() &&
         "Emitting callback function with no baton");

  QualType FuncType = FuncPointerParam.Type->getPointeeType();
  const auto *FuncProtoType = FuncType->getAs<FunctionProtoType>();
  assert(FuncProtoType && "Emitting callback with no parameter information");
  if (!FuncProtoType)
    return; // If asserts are off, we'll just fail to compile.

  std::vector<Param> CallbackParams;
  std::vector<std::string> CallbackParamsAsStrings;
  uint8_t ArgIdx = 0;
  for (QualType ParamType : FuncProtoType->param_types()) {
    Param CallbackParam;
    CallbackParam.IsFollowedByLen = false;
    CallbackParam.Type = ParamType;
    if (ParamType->isVoidPointerType())
      CallbackParam.Name = "baton";
    else
      CallbackParam.Name = "arg" + std::to_string(ArgIdx++);

    CallbackParams.push_back(CallbackParam);
    CallbackParamsAsStrings.push_back(ParamType.getAsString(method.Policy) +
                                      " " + CallbackParam.Name);
  }
  const std::string CallbackReturnTypeName =
      FuncProtoType->getReturnType().getAsString(method.Policy);
  const std::string CallbackName = method.MangledName + "_callback";

  // Emit Function Header
  std::string Header;
  llvm::raw_string_ostream HeaderStream(Header);
  HeaderStream << "static " << CallbackReturnTypeName << " " << CallbackName
               << "(" << llvm::join(CallbackParamsAsStrings, ", ") << ") {";
  EmitLine(Header);
  IndentLevel++;

  // Emit Function Body
  EmitLine("// RPC connection setup and sanity checking");
  EmitLine("CallbackInfo *callback_info = (CallbackInfo *)baton;");
  EmitLine("rpc_common::ConnectionSP connection_sp = "
           "rpc_common::Connection::GetConnectionFromID(callback_info->"
           "connection_id);");
  EmitLine("if (!connection_sp)");
  IndentLevel++;
  if (FuncProtoType->getReturnType()->isVoidType())
    EmitLine("return;");
  else
    EmitLine("return {};");
  IndentLevel--;

  EmitLine("// Preparing to make the call");
  EmitLine("static RPCFunctionInfo g_func(\"" + CallbackName + "\");");
  EmitLine("RPCStream send;");
  EmitLine("RPCStream response;");
  EmitLine("g_func.Encode(send);");

  EmitLine("// The first thing we encode is the callback address so that the "
           "client-side can know where the callback is");
  EmitLine("RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, "
           "callback_info->callback);");
  EmitLine("// Encode all the arguments");
  for (const Param &CallbackParam : CallbackParams) {
    if (lldb_rpc_gen::TypeIsSBClass(CallbackParam.Type)) {

      // FIXME: SB class server references are stored as non-const references so
      // that we can actually change them as needed. If a parameter is marked
      // const, we will fail to compile because we cannot make an
      // SBFooServerReference from a `const SBFoo &`.
      // To work around this issue, we'll apply a `const_cast` if needed so we
      // can continue to generate callbacks for now, but we really should
      // rethink the way we store object IDs server-side to support
      // const-qualified parameters.
      QualType UnderlyingSBClass =
          lldb_rpc_gen::GetUnderlyingType(CallbackParam.Type);
      QualType UnqualifiedUnderlyingSBClass =
          UnderlyingSBClass.getUnqualifiedType();

      std::string SBClassName = GetSBClassNameFromType(UnderlyingSBClass);
      llvm::StringRef SBClassNameRef(SBClassName);
      SBClassNameRef.consume_front("lldb::");

      std::string ServerReferenceLine;
      llvm::raw_string_ostream ServerReferenceLineStream(ServerReferenceLine);
      ServerReferenceLineStream << "rpc_server::" << SBClassNameRef
                                << "ServerReference " << CallbackParam.Name
                                << "_ref(";

      if (UnderlyingSBClass.isConstQualified()) {
        QualType NonConstSBType =
            method.Context.getLValueReferenceType(UnqualifiedUnderlyingSBClass);
        ServerReferenceLineStream << "const_cast<" << NonConstSBType << ">(";
      }
      ServerReferenceLineStream << CallbackParam.Name;
      if (UnderlyingSBClass.isConstQualified())
        ServerReferenceLineStream << ")";

      ServerReferenceLineStream << ");";
      EmitLine(ServerReferenceLine);
      EmitLine(
          CallbackParam.Name +
          "_ref.Encode(send, rpc_common::RPCPacket::ValueType::Argument);");
    } else {
      std::string ParamName;
      if (CallbackParam.Type->isVoidPointerType())
        ParamName = "callback_info->baton";
      else
        ParamName = CallbackParam.Name;
      EmitLine(
          "RPCValueEncoder(send, rpc_common::RPCPacket::ValueType::Argument, " +
          ParamName + ");");
    }
  }

  if (!FuncProtoType->getReturnType()->isVoidType()) {
    EmitLine("// Storage for return value");
    const bool ReturnsSBClass =
        lldb_rpc_gen::TypeIsSBClass(FuncProtoType->getReturnType());
    std::string ReturnValueLine = CallbackReturnTypeName;
    llvm::raw_string_ostream ReturnValueLineStream(ReturnValueLine);

    if (ReturnsSBClass)
      ReturnValueLineStream << " *";
    ReturnValueLineStream << " __result = ";
    if (ReturnsSBClass)
      ReturnValueLineStream << "nullptr";
    else
      ReturnValueLineStream << "{}";
    ReturnValueLineStream << ";";
    EmitLine(ReturnValueLine);
  }

  EmitLine(
      "if (connection_sp->SendRPCCallAndWaitForResponse(send, response)) {");
  IndentLevel++;
  if (!FuncProtoType->getReturnType()->isVoidType()) {
    if (lldb_rpc_gen::TypeIsSBClass(FuncProtoType->getReturnType())) {
      EmitLine("__result = rpc_server::RPCServerObjectDecoder<" +
               CallbackReturnTypeName +
               ">(response, rpc_common::RPCPacket::ValueType::ReturnValue);");
    } else
      EmitLine("RPCValueDecoder(response, "
               "rpc_common::RPCPacket::ValueType::ReturnValue, __result);");
  }
  IndentLevel--;
  EmitLine("}");
  if (!FuncProtoType->getReturnType()->isVoidType()) {
    if (lldb_rpc_gen::TypeIsSBClass(FuncProtoType->getReturnType()))
      EmitLine("return *__result;");
    else
      EmitLine("return __result;");
  }

  // Emit Function Footer;
  IndentLevel--;
  EmitLine("};");
}
