//===-- Internalize.cpp - Mark functions internal -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass loops over all of the functions in the input module, looking for a
// main function.  If a main function is found, all other functions and all
// global variables with initializers are marked as internal.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "internalize"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/Statistic.h"
#include <fstream>
#include <set>
using namespace llvm;

STATISTIC(NumAliases  , "Number of aliases internalized");
STATISTIC(NumFunctions, "Number of functions internalized");
STATISTIC(NumGlobals  , "Number of global vars internalized");

// APIFile - A file which contains a list of symbols that should not be marked
// external.
static cl::opt<std::string>
APIFile("internalize-public-api-file", cl::value_desc("filename"),
        cl::desc("A file containing list of symbol names to preserve"));

// APIList - A list of symbols that should not be marked internal.
static cl::list<std::string>
APIList("internalize-public-api-list", cl::value_desc("list"),
        cl::desc("A list of symbol names to preserve"),
        cl::CommaSeparated);

namespace {
  class VISIBILITY_HIDDEN InternalizePass : public ModulePass {
    std::set<std::string> ExternalNames;
    /// If no api symbols were specified and a main function is defined,
    /// assume the main function is the only API
    bool AllButMain;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit InternalizePass(bool AllButMain = true);
    explicit InternalizePass(const std::vector <const char *>& exportList);
    void LoadFile(const char *Filename);
    virtual bool runOnModule(Module &M);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      AU.addPreserved<CallGraph>();
    }
  };
} // end anonymous namespace

char InternalizePass::ID = 0;
static RegisterPass<InternalizePass>
X("internalize", "Internalize Global Symbols");

InternalizePass::InternalizePass(bool AllButMain)
  : ModulePass(&ID), AllButMain(AllButMain){
  if (!APIFile.empty())           // If a filename is specified, use it.
    LoadFile(APIFile.c_str());
  if (!APIList.empty())           // If a list is specified, use it as well.
    ExternalNames.insert(APIList.begin(), APIList.end());
}

InternalizePass::InternalizePass(const std::vector<const char *>&exportList)
  : ModulePass(&ID), AllButMain(false){
  for(std::vector<const char *>::const_iterator itr = exportList.begin();
        itr != exportList.end(); itr++) {
    ExternalNames.insert(*itr);
  }
}

void InternalizePass::LoadFile(const char *Filename) {
  // Load the APIFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    cerr << "WARNING: Internalize couldn't load file '" << Filename
         << "'! Continuing as if it's empty.\n";
    return; // Just continue as if the file were empty
  }
  while (In) {
    std::string Symbol;
    In >> Symbol;
    if (!Symbol.empty())
      ExternalNames.insert(Symbol);
  }
}

bool InternalizePass::runOnModule(Module &M) {
  CallGraph *CG = getAnalysisIfAvailable<CallGraph>();
  CallGraphNode *ExternalNode = CG ? CG->getExternalCallingNode() : 0;

  if (ExternalNames.empty()) {
    // Return if we're not in 'all but main' mode and have no external api
    if (!AllButMain)
      return false;
    // If no list or file of symbols was specified, check to see if there is a
    // "main" symbol defined in the module.  If so, use it, otherwise do not
    // internalize the module, it must be a library or something.
    //
    Function *MainFunc = M.getFunction("main");
    if (MainFunc == 0 || MainFunc->isDeclaration())
      return false;  // No main found, must be a library...

    // Preserve main, internalize all else.
    ExternalNames.insert(MainFunc->getName());
  }

  bool Changed = false;

  // Mark all functions not in the api as internal.
  // FIXME: maybe use private linkage?
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration() &&         // Function must be defined here
        !I->hasLocalLinkage() &&  // Can't already have internal linkage
        !ExternalNames.count(I->getName())) {// Not marked to keep external?
      I->setLinkage(GlobalValue::InternalLinkage);
      // Remove a callgraph edge from the external node to this function.
      if (ExternalNode) ExternalNode->removeOneAbstractEdgeTo((*CG)[I]);
      Changed = true;
      ++NumFunctions;
      DOUT << "Internalizing func " << I->getName() << "\n";
    }

  // Never internalize the llvm.used symbol.  It is used to implement
  // attribute((used)).
  ExternalNames.insert("llvm.used");

  // Never internalize anchors used by the machine module info, else the info
  // won't find them.  (see MachineModuleInfo.)
  ExternalNames.insert("llvm.dbg.compile_units");
  ExternalNames.insert("llvm.dbg.global_variables");
  ExternalNames.insert("llvm.dbg.subprograms");
  ExternalNames.insert("llvm.global_ctors");
  ExternalNames.insert("llvm.global_dtors");
  ExternalNames.insert("llvm.noinline");
  ExternalNames.insert("llvm.global.annotations");

  // Mark all global variables with initializers that are not in the api as
  // internal as well.
  // FIXME: maybe use private linkage?
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasLocalLinkage() &&
        !ExternalNames.count(I->getName())) {
      I->setLinkage(GlobalValue::InternalLinkage);
      Changed = true;
      ++NumGlobals;
      DOUT << "Internalized gvar " << I->getName() << "\n";
    }

  // Mark all aliases that are not in the api as internal as well.
  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasInternalLinkage() &&
        !ExternalNames.count(I->getName())) {
      I->setLinkage(GlobalValue::InternalLinkage);
      Changed = true;
      ++NumAliases;
      DOUT << "Internalized alias " << I->getName() << "\n";
    }

  return Changed;
}

ModulePass *llvm::createInternalizePass(bool AllButMain) {
  return new InternalizePass(AllButMain);
}

ModulePass *llvm::createInternalizePass(const std::vector <const char *> &el) {
  return new InternalizePass(el);
}
