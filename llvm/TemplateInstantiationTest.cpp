//===- unittests/CodeGen/TemplateInstantiationTest.cpp - template instantiation test -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestCompiler.h"

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/GlobalDecl.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/CodeGen/CodeGenABITypes.h"
#include "clang/CodeGen/ModuleBuilder.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/Sema.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include "gtest/gtest.h"

#include "llvm/Analysis/CallGraph.h"
#include <unordered_set>

using namespace llvm;
using namespace clang;

namespace {

static const bool DebugThisTest = false;

// forward declarations
struct TemplateInstantiationASTConsumer;
static void test_instantiation_fns(TemplateInstantiationASTConsumer *my);
static bool test_instantiation_fns_ran;

// This forwards the calls to the Clang CodeGenerator
// so that we can test CodeGen functions while it is open.
// It accumulates toplevel decls in HandleTopLevelDecl and
// calls test_instantiation_fns() in HandleTranslationUnit
// after forwarding that function to the CodeGenerator.

struct TemplateInstantiationASTConsumer : public ASTConsumer {
  std::unique_ptr<CodeGenerator> Builder;
  std::vector<Decl*> toplevel_decls;

  TemplateInstantiationASTConsumer(std::unique_ptr<CodeGenerator> Builder_in)
    : ASTConsumer(), Builder(std::move(Builder_in))
  {
  }

  ~TemplateInstantiationASTConsumer() { }

  void Initialize(ASTContext &Context) override;
  void HandleCXXStaticMemberVarInstantiation(VarDecl *VD) override;
  bool HandleTopLevelDecl(DeclGroupRef D) override;
  void HandleInlineFunctionDefinition(FunctionDecl *D) override;
  void HandleInterestingDecl(DeclGroupRef D) override;
  void HandleTranslationUnit(ASTContext &Ctx) override;
  void HandleTagDeclDefinition(TagDecl *D) override;
  void HandleTagDeclRequiredDefinition(const TagDecl *D) override;
  void HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) override;
  void HandleTopLevelDeclInObjCContainer(DeclGroupRef D) override;
  void HandleImplicitImportDecl(ImportDecl *D) override;
  void CompleteTentativeDefinition(VarDecl *D) override;
  void AssignInheritanceModel(CXXRecordDecl *RD) override;
  void HandleVTable(CXXRecordDecl *RD) override;
  ASTMutationListener *GetASTMutationListener() override;
  ASTDeserializationListener *GetASTDeserializationListener() override;
  void PrintStats() override;
  bool shouldSkipFunctionBody(Decl *D) override;
};

void TemplateInstantiationASTConsumer::Initialize(ASTContext &Context) {
  Builder->Initialize(Context);
}

bool TemplateInstantiationASTConsumer::HandleTopLevelDecl(DeclGroupRef DG) {

  for (DeclGroupRef::iterator I = DG.begin(), E = DG.end(); I != E; ++I) {
    toplevel_decls.push_back(*I);
  }

  return Builder->HandleTopLevelDecl(DG);
}

void TemplateInstantiationASTConsumer::HandleInlineFunctionDefinition(FunctionDecl *D) {
  Builder->HandleInlineFunctionDefinition(D);
}

void TemplateInstantiationASTConsumer::HandleInterestingDecl(DeclGroupRef D) {
  Builder->HandleInterestingDecl(D);
}

void TemplateInstantiationASTConsumer::HandleTranslationUnit(ASTContext &Context) {
  // HandleTranslationUnit can close the module
  Builder->HandleTranslationUnit(Context);
  test_instantiation_fns(this);
}

void TemplateInstantiationASTConsumer::HandleTagDeclDefinition(TagDecl *D) {
  Builder->HandleTagDeclDefinition(D);
}

void TemplateInstantiationASTConsumer::HandleTagDeclRequiredDefinition(const TagDecl *D) {
  Builder->HandleTagDeclRequiredDefinition(D);
}

void TemplateInstantiationASTConsumer::HandleCXXImplicitFunctionInstantiation(FunctionDecl *D) {
  Builder->HandleCXXImplicitFunctionInstantiation(D);
}

void TemplateInstantiationASTConsumer::HandleTopLevelDeclInObjCContainer(DeclGroupRef D) {
  Builder->HandleTopLevelDeclInObjCContainer(D);
}

void TemplateInstantiationASTConsumer::HandleImplicitImportDecl(ImportDecl *D) {
  Builder->HandleImplicitImportDecl(D);
}

void TemplateInstantiationASTConsumer::CompleteTentativeDefinition(VarDecl *D) {
  Builder->CompleteTentativeDefinition(D);
}

void TemplateInstantiationASTConsumer::AssignInheritanceModel(CXXRecordDecl *RD) {
  Builder->AssignInheritanceModel(RD);
}

void TemplateInstantiationASTConsumer::HandleCXXStaticMemberVarInstantiation(VarDecl *VD) {
   Builder->HandleCXXStaticMemberVarInstantiation(VD);
}

void TemplateInstantiationASTConsumer::HandleVTable(CXXRecordDecl *RD) {
   Builder->HandleVTable(RD);
 }

ASTMutationListener *TemplateInstantiationASTConsumer::GetASTMutationListener() {
  return Builder->GetASTMutationListener();
}

ASTDeserializationListener *TemplateInstantiationASTConsumer::GetASTDeserializationListener() {
  return Builder->GetASTDeserializationListener();
}

void TemplateInstantiationASTConsumer::PrintStats() {
  Builder->PrintStats();
}

bool TemplateInstantiationASTConsumer::shouldSkipFunctionBody(Decl *D) {
  return Builder->shouldSkipFunctionBody(D);
}

const char TestProgram[] = "struct base { public : base() {} template <typename T> base(T x) {} }; struct derived : public base { public: derived() {} derived(derived& that): base(that) {} }; int main() { derived d1; derived d2 = d1; return 0;}";

bool hasCycles(const Function *CurrentFunction,
               std::unordered_set<const Function *> &VisitedFunctions,
               std::unordered_set<const Function *> &RecursionStack,
               const CallGraphNode* CurrentNode) {
  VisitedFunctions.insert(CurrentFunction);
  RecursionStack.insert(CurrentFunction);
  for (CallGraphNode::const_iterator IT = CurrentNode->begin(), END = CurrentNode->end(); IT != END; ++IT) {
    if (const Function *CalleeFunction = IT->second->getFunction()) {
      if (RecursionStack.count(CalleeFunction)) {
        return true;
      }
      if (VisitedFunctions.count(CalleeFunction) == 0 && hasCycles(CalleeFunction, VisitedFunctions, RecursionStack, IT->second)) {
        return true;
      }
    }
  }
  RecursionStack.erase(CurrentFunction);
  return false;
}

static void test_instantiation_fns(TemplateInstantiationASTConsumer *InstantiationASTConsumer) {
  test_instantiation_fns_ran = true;
  llvm::Module* Mymodule = InstantiationASTConsumer->Builder->GetModule();
  CallGraph MyCallGraph(*Mymodule);
  std::unordered_set<const Function *> VisitedFunctions;
  std::unordered_set<const Function *> RecursionStack;
  for (llvm::CallGraph::const_iterator IT = MyCallGraph.begin(), END = MyCallGraph.end();
       IT != END; ++IT) {
    const Function* MyFunction = IT->first;
    const CallGraphNode* MyCallGraphNode = IT->second.get();
    if (MyFunction && VisitedFunctions.count(MyFunction) == 0){
      if(hasCycles(MyFunction, VisitedFunctions, RecursionStack, MyCallGraphNode)) {
        test_instantiation_fns_ran = false;
        break;
      }
    }
  }
}
 
TEST(BaseConstructorTemplateInstantiationTest, BaseConstructorTemplateInstantiationTest) {
  clang::LangOptions LO;
  LO.CPlusPlus = 1;
  TestCompiler Compiler(LO);
  auto CustomASTConsumer
    = std::make_unique<TemplateInstantiationASTConsumer>(std::move(Compiler.CG));

  Compiler.init(TestProgram, std::move(CustomASTConsumer));
  ParseAST(Compiler.compiler.getSema(), false, false);

  ASSERT_TRUE(test_instantiation_fns_ran);
}

} // end anonymous namespace

