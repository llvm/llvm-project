#include "RPCBindingsHarnessEmitter.h"
#include "RPCClientCallbacksSourceEmitter.h"
#include "RPCCommon.h"
#include "RPCLibraryHeaderEmitter.h"
#include "RPCLibrarySourceEmitter.h"
#include "RPCServerHeaderEmitter.h"
#include "RPCServerSourceEmitter.h"

#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/SourceManager.h"
#include "clang/CodeGen/ObjectFilePCHContainerWriter.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Serialization/ObjectFilePCHContainerReader.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory RPCGenCategory("Tool for generating LLDBRPC");

static llvm::cl::opt<std::string>
    OutputDir("output-dir",
              llvm::cl::desc("Directory to output generated files to"),
              llvm::cl::init(""), llvm::cl::cat(RPCGenCategory));

static std::string GetLibraryOutputDirectory() {
  llvm::SmallString<128> Path(OutputDir.getValue());
  llvm::sys::path::append(Path, "lib");
  return std::string(Path);
}

static std::string GetServerOutputDirectory() {
  llvm::SmallString<128> Path(OutputDir.getValue());
  llvm::sys::path::append(Path, "server");
  return std::string(Path);
}

static std::unique_ptr<llvm::ToolOutputFile>
CreateOutputFile(llvm::StringRef OutputDir, llvm::StringRef Filename) {
  llvm::SmallString<128> Path(OutputDir);
  llvm::sys::path::append(Path, Filename);

  std::error_code EC;
  auto OutputFile =
      std::make_unique<llvm::ToolOutputFile>(Path, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Failed to create output file: " << Path << "!\n";
    return nullptr;
  }
  return OutputFile;
}

struct GeneratedByproducts {
  std::set<std::string> ClassNames;
  std::set<std::string> MangledMethodNames;
  std::set<std::string> SkippedMethodNames;
  std::set<lldb_rpc_gen::Method> CallbackMethods;
};

enum SupportLevel {
  eUnsupported,
  eUnimplemented,
  eImplemented,
};

class SBVisitor : public RecursiveASTVisitor<SBVisitor> {
public:
  SBVisitor(
      GeneratedByproducts &Byproducts, SourceManager &Manager,
      ASTContext &Context,
      std::unique_ptr<llvm::ToolOutputFile> &&ServerMethodOutputFile,
      std::unique_ptr<llvm::ToolOutputFile> &&ServerHeaderOutputFile,
      std::unique_ptr<llvm::ToolOutputFile> &&LibrarySourceOutputFile,
      std::unique_ptr<llvm::ToolOutputFile> &&LibraryHeaderOutputFile,
      lldb_rpc_gen::RPCClientCallbacksSourceEmitter &UserClientSourceEmitter,
      lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter)
      : Byproducts(Byproducts), Manager(Manager), Context(Context),
        ServerSourceEmitter(std::move(ServerMethodOutputFile)),
        ServerHeaderEmitter(std::move(ServerHeaderOutputFile)),
        LibrarySourceEmitter(std::move(LibrarySourceOutputFile)),
        LibraryHeaderEmitter(std::move(LibraryHeaderOutputFile)),
        ClientCallbacksSourceEmitter(UserClientSourceEmitter),
        BindingsHarnessEmitter(BindingsHarnessEmitter) {}

  ~SBVisitor() {}

  bool VisitCXXRecordDecl(CXXRecordDecl *RDecl) {
    if (ShouldSkipRecord(RDecl))
      return true;

    const std::string ClassName = RDecl->getNameAsString();
    Byproducts.ClassNames.insert(ClassName);

    // Print 'bool' instead of '_Bool'.
    PrintingPolicy Policy(Context.getLangOpts());
    Policy.Bool = true;

    LibraryHeaderEmitter.StartClass(ClassName);
    LibrarySourceEmitter.StartClass(ClassName);
    BindingsHarnessEmitter.StartClass(ClassName);
    for (Decl *D : RDecl->decls())
      if (auto *E = dyn_cast_or_null<EnumDecl>(D))
        LibraryHeaderEmitter.EmitEnum(E);

    for (CXXMethodDecl *MDecl : RDecl->methods()) {
      const std::string MangledName =
          lldb_rpc_gen::GetMangledName(Context, MDecl);
      const bool IsDisallowed = lldb_rpc_gen::MethodIsDisallowed(MangledName);
      const bool HasCallbackParameter =
          lldb_rpc_gen::HasCallbackParameter(MDecl);
      SupportLevel MethodSupportLevel = GetMethodSupportLevel(MDecl);
      if (MethodSupportLevel == eImplemented && !IsDisallowed) {
        const lldb_rpc_gen::Method Method(MDecl, Policy, Context);
        ServerSourceEmitter.EmitMethod(Method);
        ServerHeaderEmitter.EmitMethod(Method);
        LibrarySourceEmitter.EmitMethod(Method);
        LibraryHeaderEmitter.EmitMethod(Method);
        BindingsHarnessEmitter.EmitMethod(Method);
        Byproducts.MangledMethodNames.insert(MangledName);
        if (HasCallbackParameter) {
          ClientCallbacksSourceEmitter.EmitMethod(Method);
          Byproducts.CallbackMethods.insert(Method);
        }
      } else if (MethodSupportLevel == eUnimplemented)
        Byproducts.SkippedMethodNames.insert(MangledName);
    }
    LibraryHeaderEmitter.EndClass();
    LibrarySourceEmitter.EndClass();
    BindingsHarnessEmitter.EndClass();
    return true;
  }

private:
  /// Determines whether we should skip a RecordDecl.
  /// Conditions for skipping:
  ///   - Anything not in the header itself
  ///   - Certain inconvenient classes
  ///   - Records without definitions (forward declarations)
  bool ShouldSkipRecord(CXXRecordDecl *Decl) {
    const Type *DeclType = Decl->getTypeForDecl();
    QualType CanonicalType = DeclType->getCanonicalTypeInternal();
    return !Manager.isInMainFile(Decl->getBeginLoc()) ||
           !Decl->hasDefinition() || Decl->getDefinition() != Decl ||
           lldb_rpc_gen::TypeIsDisallowedClass(CanonicalType);
  }

  /// Check the support level for a type
  /// Known unsupported types:
  ///  - FILE * (We do not want to expose this primitive)
  ///  - Types that are internal to LLDB
  SupportLevel GetTypeSupportLevel(QualType Type) {
    const std::string TypeName = Type.getAsString();
    if (TypeName == "FILE *" || lldb_rpc_gen::TypeIsFromLLDBPrivate(Type))
      return eUnsupported;

    if (lldb_rpc_gen::TypeIsDisallowedClass(Type))
      return eUnsupported;

    return eImplemented;
  }

  /// Determine the support level of a given method.
  /// Known unsupported methods:
  ///   - Non-public methods (lldb-rpc is a client and can only see public
  ///     things)
  ///   - Copy assignment operators (the client side will handle this)
  ///   - Move assignment operators (the client side will handle this)
  ///   - Methods involving unsupported types.
  /// Known unimplemented methods:
  ///   - No variadic functions, e.g. Printf
  SupportLevel GetMethodSupportLevel(CXXMethodDecl *MDecl) {
    AccessSpecifier AS = MDecl->getAccess();
    if (AS != AccessSpecifier::AS_public)
      return eUnsupported;
    if (MDecl->isCopyAssignmentOperator())
      return eUnsupported;
    if (MDecl->isMoveAssignmentOperator())
      return eUnsupported;

    if (MDecl->isVariadic())
      return eUnimplemented;

    SupportLevel ReturnTypeLevel = GetTypeSupportLevel(MDecl->getReturnType());
    if (ReturnTypeLevel != eImplemented)
      return ReturnTypeLevel;

    for (auto *ParamDecl : MDecl->parameters()) {
      SupportLevel ParamTypeLevel = GetTypeSupportLevel(ParamDecl->getType());
      if (ParamTypeLevel != eImplemented)
        return ParamTypeLevel;
    }

    // FIXME: If a callback does not take a `void *baton` parameter, it is
    // considered unsupported at this time. On the server-side, we hijack the
    // baton argument in order to pass additional information to the server-side
    // callback so we can correctly perform a reverse RPC call back to the
    // client. Without this baton, we would need the server-side callback to
    // have some side channel by which it obtained that information, and
    // spending time designing that doesn't outweight the cost of doing it at
    // the moment.
    bool HasCallbackParameter = false;
    bool HasBatonParameter = false;
    auto End = MDecl->parameters().end();
    for (auto Iter = MDecl->parameters().begin(); Iter != End; Iter++) {
      if ((*Iter)->getType()->isFunctionPointerType()) {
        HasCallbackParameter = true;
        continue;
      }

      // FIXME: We assume that if we have a function pointer and a void pointer
      // together in the same parameter list, that it is not followed by a
      // length argument. If that changes, we will need to revisit this
      // implementation.
      if ((*Iter)->getType()->isVoidPointerType())
        HasBatonParameter = true;
    }

    if (HasCallbackParameter && !HasBatonParameter)
      return eUnimplemented;

    return eImplemented;
  }

  GeneratedByproducts &Byproducts;
  SourceManager &Manager;
  ASTContext &Context;
  lldb_rpc_gen::RPCServerSourceEmitter ServerSourceEmitter;
  lldb_rpc_gen::RPCServerHeaderEmitter ServerHeaderEmitter;
  lldb_rpc_gen::RPCLibrarySourceEmitter LibrarySourceEmitter;
  lldb_rpc_gen::RPCLibraryHeaderEmitter LibraryHeaderEmitter;
  lldb_rpc_gen::RPCClientCallbacksSourceEmitter &ClientCallbacksSourceEmitter;
  lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter;
};

class SBConsumer : public ASTConsumer {
public:
  SBConsumer(GeneratedByproducts &Byproducts, SourceManager &Manager,
             ASTContext &Context,
             std::unique_ptr<llvm::ToolOutputFile> &&ServerMethodOutputFile,
             std::unique_ptr<llvm::ToolOutputFile> &&ServerHeaderOutputFile,
             std::unique_ptr<llvm::ToolOutputFile> &&LibrarySourceOutputFile,
             std::unique_ptr<llvm::ToolOutputFile> &&LibraryHeaderOutputFile,
             lldb_rpc_gen::RPCClientCallbacksSourceEmitter
                 &ClientCallbacksSourceEmitter,
             lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter)
      : Visitor(Byproducts, Manager, Context, std::move(ServerMethodOutputFile),
                std::move(ServerHeaderOutputFile),
                std::move(LibrarySourceOutputFile),
                std::move(LibraryHeaderOutputFile),
                ClientCallbacksSourceEmitter, BindingsHarnessEmitter) {}
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    for (Decl *D : DR)
      Visitor.TraverseDecl(D);

    return true;
  }

private:
  SBVisitor Visitor;
};

class SBAction : public ASTFrontendAction {
public:
  SBAction(
      GeneratedByproducts &Byproducts,
      lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter,
      lldb_rpc_gen::RPCClientCallbacksSourceEmitter &UserClientSourceEmitter)
      : Byproducts(Byproducts), BindingsHarnessEmitter(BindingsHarnessEmitter),
        ClientCallbacksSourceEmitter(UserClientSourceEmitter) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, llvm::StringRef File) override {
    llvm::StringRef FilenameNoExt =
        llvm::sys::path::stem(llvm::sys::path::filename(File));

    const std::string ServerMethodFilename =
        "Server_" + FilenameNoExt.str() + ".cpp";
    std::unique_ptr<llvm::ToolOutputFile> ServerMethodOutputFile =
        CreateOutputFile(GetServerOutputDirectory(), ServerMethodFilename);
    if (!ServerMethodOutputFile)
      return nullptr;

    const std::string ServerHeaderFilename =
        "Server_" + FilenameNoExt.str() + ".h";
    std::unique_ptr<llvm::ToolOutputFile> ServerHeaderOutputFile =
        CreateOutputFile(GetServerOutputDirectory(), ServerHeaderFilename);
    if (!ServerHeaderOutputFile)
      return nullptr;

    const std::string LibrarySourceFilename = FilenameNoExt.str() + ".cpp";
    std::unique_ptr<llvm::ToolOutputFile> LibrarySourceOutputFile =
        CreateOutputFile(GetLibraryOutputDirectory(), LibrarySourceFilename);
    if (!LibrarySourceOutputFile)
      return nullptr;

    const std::string LibraryHeaderFilename = FilenameNoExt.str() + ".h";
    std::unique_ptr<llvm::ToolOutputFile> LibraryHeaderOutputFile =
        CreateOutputFile(GetLibraryOutputDirectory(), LibraryHeaderFilename);
    if (!LibraryHeaderOutputFile)
      return nullptr;

    ServerMethodOutputFile->keep();
    ServerHeaderOutputFile->keep();
    LibrarySourceOutputFile->keep();
    LibraryHeaderOutputFile->keep();
    return std::make_unique<SBConsumer>(
        Byproducts, CI.getSourceManager(), CI.getASTContext(),
        std::move(ServerMethodOutputFile), std::move(ServerHeaderOutputFile),
        std::move(LibrarySourceOutputFile), std::move(LibraryHeaderOutputFile),
        ClientCallbacksSourceEmitter, BindingsHarnessEmitter);
  }

private:
  GeneratedByproducts &Byproducts;
  lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter;
  lldb_rpc_gen::RPCClientCallbacksSourceEmitter &ClientCallbacksSourceEmitter;
};

class SBActionFactory : public FrontendActionFactory {
public:
  SBActionFactory(
      GeneratedByproducts &Byproducts,
      lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter,
      lldb_rpc_gen::RPCClientCallbacksSourceEmitter
          &ClientCallbacksSourceEmitter)
      : Byproducts(Byproducts), BindingsHarnessEmitter(BindingsHarnessEmitter),
        ClientCallbacksSourceEmitter(ClientCallbacksSourceEmitter) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<SBAction>(Byproducts, BindingsHarnessEmitter,
                                      ClientCallbacksSourceEmitter);
  }

private:
  GeneratedByproducts &Byproducts;
  lldb_rpc_gen::RPCBindingsHarnessEmitter &BindingsHarnessEmitter;
  lldb_rpc_gen::RPCClientCallbacksSourceEmitter &ClientCallbacksSourceEmitter;
};

bool EmitAmalgamatedServerHeader(const std::vector<std::string> &Files) {
  // Create the file
  static constexpr llvm::StringLiteral AmalgamatedServerHeaderName = "SBAPI.h";
  std::unique_ptr<llvm::ToolOutputFile> AmalgamatedServerHeader =
      CreateOutputFile(GetServerOutputDirectory(), AmalgamatedServerHeaderName);
  if (!AmalgamatedServerHeader)
    return false;

  // Write the header
  AmalgamatedServerHeader->os()
      << "#ifndef GENERATED_LLDB_RPC_SERVER_SBAPI_H\n";
  AmalgamatedServerHeader->os()
      << "#define GENERATED_LLDB_RPC_SERVER_SBAPI_H\n";
  for (const auto &File : Files) {
    llvm::StringRef FilenameNoExt =
        llvm::sys::path::stem(llvm::sys::path::filename(File));
    const std::string ServerHeaderFilename =
        "Server_" + FilenameNoExt.str() + ".h";

    AmalgamatedServerHeader->os()
        << "#include \"" + ServerHeaderFilename + "\"\n";
  }
  AmalgamatedServerHeader->os() << "#include \"SBAPIExtensions.h\"\n";
  AmalgamatedServerHeader->os()
      << "#endif // GENERATED_LLDB_RPC_SERVER_SBAPI_H\n";
  AmalgamatedServerHeader->keep();
  return true;
}

bool EmitAmalgamatedLibraryHeader(const std::vector<std::string> &Files) {
  static constexpr llvm::StringLiteral AmalgamatedLibraryHeaderName =
      "LLDBRPC.h";
  std::unique_ptr<llvm::ToolOutputFile> AmalgamatedLibraryHeader =
      CreateOutputFile(GetLibraryOutputDirectory(),
                       AmalgamatedLibraryHeaderName);
  if (!AmalgamatedLibraryHeader)
    return false;

  AmalgamatedLibraryHeader->os() << "#ifndef LLDBRPC_H\n";
  AmalgamatedLibraryHeader->os() << "#define LLDBRPC_H\n";
  AmalgamatedLibraryHeader->os() << "#include \"SBLanguages.h\"\n";
  for (const auto &File : Files) {
    llvm::StringRef Filename = llvm::sys::path::filename(File);
    AmalgamatedLibraryHeader->os() << "#include \"" << Filename << "\"\n";
  }

  AmalgamatedLibraryHeader->os() << "#endif // LLDBRPC_H\n";
  AmalgamatedLibraryHeader->keep();
  return true;
}

bool EmitClassNamesFile(std::set<std::string> &ClassNames) {
  static constexpr llvm::StringLiteral ClassNamesFileName = "SBClasses.def";
  std::unique_ptr<llvm::ToolOutputFile> ClassNamesFile =
      CreateOutputFile(OutputDir.getValue(), ClassNamesFileName);
  if (!ClassNamesFile)
    return false;

  ClassNamesFile->os() << "#ifndef SBCLASS\n";
  ClassNamesFile->os() << "#error \"SBClass must be defined\"\n";
  ClassNamesFile->os() << "#endif\n";

  for (const auto &ClassName : ClassNames) {
    if (ClassName == "SBStream" || ClassName == "SBProgress")
      ClassNamesFile->os() << "#if !defined(SBCLASS_EXCLUDE_NONCOPYABLE)\n";
    else if (ClassName == "SBReproducer")
      ClassNamesFile->os() << "#if !defined(SBCLASS_EXCLUDE_STATICONLY)\n";

    ClassNamesFile->os() << "SBCLASS(" << ClassName << ")\n";
    if (ClassName == "SBStream" || ClassName == "SBReproducer" ||
        ClassName == "SBProgress")
      ClassNamesFile->os() << "#endif\n";
  }
  ClassNamesFile->keep();
  return true;
}

bool EmitMethodNamesFile(std::set<std::string> &MangledMethodNames) {
  static constexpr llvm::StringLiteral MethodNamesFileName = "SBAPI.def";
  std::unique_ptr<llvm::ToolOutputFile> MethodNamesFile =
      CreateOutputFile(OutputDir.getValue(), MethodNamesFileName);
  if (!MethodNamesFile)
    return false;

  MethodNamesFile->os() << "#ifndef GENERATE_SBAPI\n";
  MethodNamesFile->os() << "#error \"GENERATE_SBAPI must be defined\"\n";
  MethodNamesFile->os() << "#endif\n";

  for (const auto &MangledName : MangledMethodNames) {
    MethodNamesFile->os() << "GENERATE_SBAPI(" << MangledName << ")\n";
  }
  MethodNamesFile->keep();
  return true;
}

bool EmitSkippedMethodsFile(std::set<std::string> &SkippedMethodNames) {
  static constexpr llvm::StringLiteral FileName = "SkippedMethods.txt";
  std::unique_ptr<llvm::ToolOutputFile> File =
      CreateOutputFile(OutputDir.getValue(), FileName);
  if (!File)
    return false;

  for (const auto &Skipped : SkippedMethodNames) {
    File->os() << Skipped << "\n";
  }
  File->keep();
  return true;
}

int main(int argc, const char *argv[]) {
  auto ExpectedParser = CommonOptionsParser::create(
      argc, argv, RPCGenCategory, llvm::cl::OneOrMore,
      "Tool for generating LLDBRPC interfaces and implementations");

  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }

  if (OutputDir.empty()) {
    llvm::errs() << "Please specify an output directory for the generated "
                    "files with --output-dir!\n";
    return 1;
  }

  CommonOptionsParser &OP = ExpectedParser.get();
  auto PCHOpts = std::make_shared<PCHContainerOperations>();
  PCHOpts->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOpts->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  ClangTool T(OP.getCompilations(), OP.getSourcePathList(), PCHOpts);

  if (!EmitAmalgamatedServerHeader(OP.getSourcePathList())) {
    llvm::errs() << "Failed to create amalgamated server header\n";
    return 1;
  }

  if (!EmitAmalgamatedLibraryHeader(OP.getSourcePathList())) {
    llvm::errs() << "Failed to create amalgamated library header\n";
    return 1;
  }

  GeneratedByproducts Byproducts;

  constexpr llvm::StringLiteral BindingsHarnessFilename = "lldb.py";
  std::unique_ptr<llvm::ToolOutputFile> BindingsHarnessOutputFile =
      CreateOutputFile(OutputDir.getValue(), BindingsHarnessFilename);

  if (!BindingsHarnessOutputFile) {
    llvm::errs() << "Failed to create the bindings harness file\n";
    return 1;
  }
  BindingsHarnessOutputFile->keep();
  lldb_rpc_gen::RPCBindingsHarnessEmitter BindingsHarnessEmitter(
      std::move(BindingsHarnessOutputFile));

  static constexpr llvm::StringLiteral FileName = "RPCClientSideCallbacks.cpp";
  std::unique_ptr<llvm::ToolOutputFile> &&UserClientSourceOutputFile =
      CreateOutputFile(GetLibraryOutputDirectory(), FileName);
  if (!UserClientSourceOutputFile) {
    llvm::errs() << "Failed to create the user client callbacks source file\n";
    return 1;
  }

  UserClientSourceOutputFile->keep();
  lldb_rpc_gen::RPCClientCallbacksSourceEmitter ClientCallbacksSourceEmitter(
      std::move(UserClientSourceOutputFile));

  SBActionFactory Factory(Byproducts, BindingsHarnessEmitter,
                          ClientCallbacksSourceEmitter);
  auto Result = T.run(&Factory);
  ClientCallbacksSourceEmitter.EmitRPCClientInitialize(
      Byproducts.CallbackMethods);
  if (!EmitClassNamesFile(Byproducts.ClassNames)) {
    llvm::errs() << "Failed to create SB Class file\n";
    return 1;
  }
  if (!EmitMethodNamesFile(Byproducts.MangledMethodNames)) {
    llvm::errs() << "Failed to create Method Names file\n";
    return 1;
  }
  if (!EmitSkippedMethodsFile(Byproducts.SkippedMethodNames)) {
    llvm::errs() << "Failed to create Skipped Methods file\n";
    return 1;
  }

  return Result;
}
