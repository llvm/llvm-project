//===-- Internalize.cpp - Mark functions internal -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass loops over all of the functions and variables in the input module.
// If the function or variable is not in the list of external names given to
// the pass it is marked as internal.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "internalize"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Pass.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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
  class InternalizePass : public ModulePass {
    std::set<std::string> ExternalNames;
  public:
    static char ID; // Pass identification, replacement for typeid
    explicit InternalizePass();
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
INITIALIZE_PASS(InternalizePass, "internalize",
                "Internalize Global Symbols", false, false)

InternalizePass::InternalizePass()
  : ModulePass(ID) {
  initializeInternalizePassPass(*PassRegistry::getPassRegistry());
  if (!APIFile.empty())           // If a filename is specified, use it.
    LoadFile(APIFile.c_str());
  if (!APIList.empty())           // If a list is specified, use it as well.
    ExternalNames.insert(APIList.begin(), APIList.end());
}

InternalizePass::InternalizePass(const std::vector<const char *>&exportList)
  : ModulePass(ID){
  initializeInternalizePassPass(*PassRegistry::getPassRegistry());
  for(std::vector<const char *>::const_iterator itr = exportList.begin();
        itr != exportList.end(); itr++) {
    ExternalNames.insert(*itr);
  }
}

void InternalizePass::LoadFile(const char *Filename) {
  // Load the APIFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: Internalize couldn't load file '" << Filename
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
  bool Changed = false;

  // Never internalize functions which code-gen might insert.
  // FIXME: We should probably add this (and the __stack_chk_guard) via some
  // type of call-back in CodeGen.
  ExternalNames.insert("__stack_chk_fail");

  // Mark all functions not in the api as internal.
  // FIXME: maybe use private linkage?
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration() &&         // Function must be defined here
        // Available externally is really just a "declaration with a body".
        !I->hasAvailableExternallyLinkage() &&
        !I->hasLocalLinkage() &&  // Can't already have internal linkage
        !ExternalNames.count(I->getName())) {// Not marked to keep external?
      I->setLinkage(GlobalValue::InternalLinkage);
      // Remove a callgraph edge from the external node to this function.
      if (ExternalNode) ExternalNode->removeOneAbstractEdgeTo((*CG)[I]);
      Changed = true;
      ++NumFunctions;
      DEBUG(dbgs() << "Internalizing func " << I->getName() << "\n");
    }

  // Never internalize the llvm.used symbol.  It is used to implement
  // attribute((used)).
  // FIXME: Shouldn't this just filter on llvm.metadata section??
  ExternalNames.insert("llvm.used");
  ExternalNames.insert("llvm.compiler.used");

  // Never internalize anchors used by the machine module info, else the info
  // won't find them.  (see MachineModuleInfo.)
  ExternalNames.insert("llvm.global_ctors");
  ExternalNames.insert("llvm.global_dtors");
  ExternalNames.insert("llvm.global.annotations");

  // Never internalize symbols code-gen inserts.
  ExternalNames.insert("__stack_chk_guard");

  // Mark all global variables with initializers that are not in the api as
  // internal as well.
  // FIXME: maybe use private linkage?
  for (Module::global_iterator I = M.global_begin(), E = M.global_end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasLocalLinkage() &&
        // Available externally is really just a "declaration with a body".
        !I->hasAvailableExternallyLinkage() &&
        !ExternalNames.count(I->getName())) {
      I->setLinkage(GlobalValue::InternalLinkage);
      Changed = true;
      ++NumGlobals;
      DEBUG(dbgs() << "Internalized gvar " << I->getName() << "\n");
    }

  // Mark all aliases that are not in the api as internal as well.
  for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end();
       I != E; ++I)
    if (!I->isDeclaration() && !I->hasInternalLinkage() &&
        // Available externally is really just a "declaration with a body".
        !I->hasAvailableExternallyLinkage() &&
        !ExternalNames.count(I->getName())) {
      I->setLinkage(GlobalValue::InternalLinkage);
      Changed = true;
      ++NumAliases;
      DEBUG(dbgs() << "Internalized alias " << I->getName() << "\n");
    }

  return Changed;
}

ModulePass *llvm::createInternalizePass() {
  return new InternalizePass();
}

ModulePass *llvm::createInternalizePass(const std::vector <const char *> &el) {
  return new InternalizePass(el);
}
