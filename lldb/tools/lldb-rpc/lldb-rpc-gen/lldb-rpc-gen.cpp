//===-- lldb-rpc-gen.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RPCCommon.h"

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

static std::unique_ptr<llvm::ToolOutputFile>
CreateOutputFile(llvm::StringRef OutputDir, llvm::StringRef Filename) {
  llvm::SmallString<256> Path(OutputDir);
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
};

enum SupportLevel {
  eUnsupported,
  eUnimplemented,
  eImplemented,
};

class SBVisitor : public RecursiveASTVisitor<SBVisitor> {
public:
  SBVisitor(GeneratedByproducts &Byproducts, SourceManager &Manager,
            ASTContext &Context)
      : Byproducts(Byproducts), Manager(Manager), Context(Context) {}

  ~SBVisitor() {}

  bool VisitCXXRecordDecl(CXXRecordDecl *RDecl) {
    if (ShouldSkipRecord(RDecl))
      return true;

    const std::string ClassName = RDecl->getNameAsString();
    Byproducts.ClassNames.insert(ClassName);

    // Print 'bool' instead of '_Bool'.
    PrintingPolicy Policy(Context.getLangOpts());
    Policy.Bool = true;

    for (CXXMethodDecl *MDecl : RDecl->methods()) {
      const std::string MangledName =
          lldb_rpc_gen::GetMangledName(Context, MDecl);
      const bool IsDisallowed =
          lldb_rpc_gen::MethodIsDisallowed(Context, MDecl);
      SupportLevel MethodSupportLevel = GetMethodSupportLevel(MDecl);
      if (MethodSupportLevel == eImplemented && !IsDisallowed) {
        const lldb_rpc_gen::Method Method(MDecl, Policy, Context);
        Byproducts.MangledMethodNames.insert(MangledName);
      } else if (MethodSupportLevel == eUnimplemented)
        Byproducts.SkippedMethodNames.insert(MangledName);
    }
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
};

class SBConsumer : public ASTConsumer {
public:
  SBConsumer(GeneratedByproducts &Byproducts, SourceManager &Manager,
             ASTContext &Context)
      : Visitor(Byproducts, Manager, Context) {}
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
  SBAction(GeneratedByproducts &Byproducts) : Byproducts(Byproducts) {}

  std::unique_ptr<ASTConsumer>
  CreateASTConsumer(CompilerInstance &CI, llvm::StringRef File) override {
    llvm::StringRef FilenameNoExt =
        llvm::sys::path::stem(llvm::sys::path::filename(File));

    return std::make_unique<SBConsumer>(Byproducts, CI.getSourceManager(),
                                        CI.getASTContext());
  }

private:
  GeneratedByproducts &Byproducts;
};

class SBActionFactory : public FrontendActionFactory {
public:
  SBActionFactory(GeneratedByproducts &Byproducts) : Byproducts(Byproducts) {}

  std::unique_ptr<FrontendAction> create() override {
    return std::make_unique<SBAction>(Byproducts);
  }

private:
  GeneratedByproducts &Byproducts;
};

bool EmitClassNamesFile(std::set<std::string> &ClassNames) {
  static constexpr llvm::StringLiteral ClassNamesFileName = "SBClasses.def";
  std::unique_ptr<llvm::ToolOutputFile> ClassNamesFile =
      CreateOutputFile(OutputDir.getValue(), ClassNamesFileName);
  if (!ClassNamesFile)
    return false;

  ClassNamesFile->os() << "#ifndef SBCLASS\n"
                       << "#error \"SBClass must be defined\"\n"
                       << "#endif\n";

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

  MethodNamesFile->os() << "#ifndef GENERATE_SBAPI\n"
                        << "#error \"GENERATE_SBAPI must be defined\"\n"
                        << "#endif\n";

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

  for (const auto &Skipped : SkippedMethodNames)
    File->os() << Skipped << "\n";
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

  // Create the output directory if the user specified one does not exist.
  if (!llvm::sys::fs::exists(OutputDir.getValue())) {
    llvm::sys::fs::create_directory(OutputDir.getValue());
  }

  CommonOptionsParser &OP = ExpectedParser.get();
  auto PCHOpts = std::make_shared<PCHContainerOperations>();
  PCHOpts->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOpts->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  ClangTool T(OP.getCompilations(), OP.getSourcePathList(), PCHOpts);

  GeneratedByproducts Byproducts;

  SBActionFactory Factory(Byproducts);
  auto Result = T.run(&Factory);
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
