#include "RPCLibraryHeaderEmitter.h"
#include "RPCCommon.h"

#include "clang/AST/AST.h"
#include "clang/AST/Mangle.h"
#include "clang/Frontend/CompilerInstance.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace lldb_rpc_gen;

void RPCLibraryHeaderEmitter::StartClass(std::string ClassName) {
  CurrentClass = std::move(ClassName);
  std::string BaseClass =
      lldb_rpc_gen::SBClassInheritsFromObjectRef(CurrentClass)
          ? "ObjectRef"
          : "LocalObjectRef";
  EmitLine("class " + CurrentClass + " : public rpc::" + BaseClass + " {");
  EmitLine("public:");
  IndentLevel++;
  if (lldb_rpc_gen::SBClassRequiresDefaultCtor(CurrentClass))
    EmitLine(CurrentClass + "();");

  // NOTE: There's currently only one RPC-specific extension that is actually
  // used AFAICT. We can generalize this if we need more.
  if (CurrentClass == "SBDebugger")
    EmitLine("int SetIOFile(const char *path);");
}

void RPCLibraryHeaderEmitter::EndClass() {
  if (lldb_rpc_gen::SBClassRequiresCopyCtorAssign(CurrentClass)) {
    if (!CopyCtorEmitted)
      EmitLine(CurrentClass + "(const lldb_rpc::" + CurrentClass + " &rhs);");
    if (!CopyAssignEmitted)
      EmitLine(CurrentClass + " &operator=(const lldb_rpc::" + CurrentClass +
               " &rhs);");
  }
  if (!MoveCtorEmitted)
    EmitLine(CurrentClass + "(lldb_rpc::" + CurrentClass + " &&rhs);");
  if (!MoveAssignEmitted)
    EmitLine(CurrentClass + " &operator=(" + CurrentClass + " &&rhs);");

  IndentLevel--;
  EmitLine("}; // class " + CurrentClass);
  CurrentClass.clear();
}

void RPCLibraryHeaderEmitter::EmitMethod(const Method &method) {
  std::string DeclarationLine;
  llvm::raw_string_ostream DeclarationLineStream(DeclarationLine);

  if (method.IsExplicitCtorOrConversionMethod)
    DeclarationLineStream << "explicit ";
  else if (!method.IsInstance)
    DeclarationLineStream << "static ";

  if (!method.IsDtor && !method.IsConversionMethod && !method.IsCtor) {
    DeclarationLineStream << lldb_rpc_gen::ReplaceLLDBNamespaceWithRPCNamespace(
                                 method.ReturnType.getAsString(method.Policy))
                          << " ";
  }

  DeclarationLineStream << method.BaseName << "("
                        << method.CreateParamListAsString(
                               eLibrary, /*IncludeDefaultValue = */ true)
                        << ")";
  if (method.IsConst)
    DeclarationLineStream << " const";
  DeclarationLineStream << ";";

  EmitLine(DeclarationLine);

  if (method.IsCopyCtor)
    CopyCtorEmitted = true;
  else if (method.IsCopyAssign)
    CopyAssignEmitted = true;
  else if (method.IsMoveCtor)
    MoveCtorEmitted = true;
  else if (method.IsMoveAssign)
    MoveAssignEmitted = true;
}

void RPCLibraryHeaderEmitter::EmitEnum(EnumDecl *E) {
  // NOTE: All of the enumerations embedded in SB classes are currently
  // anonymous and backed by an unsigned int.
  EmitLine("enum : unsigned {");
  IndentLevel++;
  for (const EnumConstantDecl *EC : E->enumerators()) {
    std::string EnumValue = EC->getNameAsString();
    SmallString<16> ValueStr;
    EC->getInitVal().toString(ValueStr);
    EnumValue += " = " + ValueStr.str().str() + ", ";
    EmitLine(EnumValue);
  }

  IndentLevel--;
  EmitLine("};");
}

std::string RPCLibraryHeaderEmitter::GetHeaderGuard() {
  const std::string UpperFilenameNoExt =
      llvm::sys::path::stem(
          llvm::sys::path::filename(OutputFile->getFilename()))
          .upper();
  return "GENERATED_LLDB_RPC_LIBRARY_" + UpperFilenameNoExt + "_H";
}

void RPCLibraryHeaderEmitter::Begin() {
  const std::string HeaderGuard = GetHeaderGuard();
  EmitLine("#ifndef " + HeaderGuard);
  EmitLine("#define " + HeaderGuard);
  EmitLine("");
  EmitLine("#include <lldb-rpc/common/RPCPublic.h>");
  EmitLine("#include \"SBDefines.h\"");
  EmitLine("#include \"LLDBRPC.h\"");
  EmitLine("");
  EmitLine("namespace lldb_rpc {");
}

void RPCLibraryHeaderEmitter::End() {
  EmitLine("} // namespace lldb_rpc");
  EmitLine("#endif // " + GetHeaderGuard());
}
