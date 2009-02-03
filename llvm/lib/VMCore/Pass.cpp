//===- Pass.cpp - LLVM Pass Infrastructure Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LLVM Pass infrastructure.  It is primarily
// responsible with ensuring that passes are executed and batched together
// optimally.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/PassManager.h"
#include "llvm/Module.h"
#include "llvm/ModuleProvider.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ManagedStatic.h"
#include <algorithm>
#include <map>
#include <set>
using namespace llvm;

//===----------------------------------------------------------------------===//
// Pass Implementation
//

// Force out-of-line virtual method.
Pass::~Pass() { 
  delete Resolver; 
}

// Force out-of-line virtual method.
ModulePass::~ModulePass() { }

bool Pass::mustPreserveAnalysisID(const PassInfo *AnalysisID) const {
  return Resolver->getAnalysisIfAvailable(AnalysisID, true) != 0;
}

// dumpPassStructure - Implement the -debug-passes=Structure option
void Pass::dumpPassStructure(unsigned Offset) {
  cerr << std::string(Offset*2, ' ') << getPassName() << "\n";
}

/// getPassName - Return a nice clean name for a pass.  This usually
/// implemented in terms of the name that is registered by one of the
/// Registration templates, but can be overloaded directly.
///
const char *Pass::getPassName() const {
  if (const PassInfo *PI = getPassInfo())
    return PI->getPassName();
  return "Unnamed pass: implement Pass::getPassName()";
}

// print - Print out the internal state of the pass.  This is called by Analyze
// to print out the contents of an analysis.  Otherwise it is not necessary to
// implement this method.
//
void Pass::print(std::ostream &O,const Module*) const {
  O << "Pass::print not implemented for pass: '" << getPassName() << "'!\n";
}

// dump - call print(cerr);
void Pass::dump() const {
  print(*cerr.stream(), 0);
}

//===----------------------------------------------------------------------===//
// ImmutablePass Implementation
//
// Force out-of-line virtual method.
ImmutablePass::~ImmutablePass() { }

//===----------------------------------------------------------------------===//
// FunctionPass Implementation
//

// run - On a module, we run this pass by initializing, runOnFunction'ing once
// for every function in the module, then by finalizing.
//
bool FunctionPass::runOnModule(Module &M) {
  bool Changed = doInitialization(M);

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I)
    if (!I->isDeclaration())      // Passes are not run on external functions!
    Changed |= runOnFunction(*I);

  return Changed | doFinalization(M);
}

// run - On a function, we simply initialize, run the function, then finalize.
//
bool FunctionPass::run(Function &F) {
  // Passes are not run on external functions!
  if (F.isDeclaration()) return false;

  bool Changed = doInitialization(*F.getParent());
  Changed |= runOnFunction(F);
  return Changed | doFinalization(*F.getParent());
}

//===----------------------------------------------------------------------===//
// BasicBlockPass Implementation
//

// To run this pass on a function, we simply call runOnBasicBlock once for each
// function.
//
bool BasicBlockPass::runOnFunction(Function &F) {
  bool Changed = doInitialization(F);
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    Changed |= runOnBasicBlock(*I);
  return Changed | doFinalization(F);
}

//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//
namespace {
class PassRegistrar {
  /// PassInfoMap - Keep track of the passinfo object for each registered llvm
  /// pass.
  typedef std::map<intptr_t, const PassInfo*> MapType;
  MapType PassInfoMap;
  
  /// AnalysisGroupInfo - Keep track of information for each analysis group.
  struct AnalysisGroupInfo {
    const PassInfo *DefaultImpl;
    std::set<const PassInfo *> Implementations;
    AnalysisGroupInfo() : DefaultImpl(0) {}
  };
  
  /// AnalysisGroupInfoMap - Information for each analysis group.
  std::map<const PassInfo *, AnalysisGroupInfo> AnalysisGroupInfoMap;

public:
  
  const PassInfo *GetPassInfo(intptr_t TI) const {
    MapType::const_iterator I = PassInfoMap.find(TI);
    return I != PassInfoMap.end() ? I->second : 0;
  }
  
  void RegisterPass(const PassInfo &PI) {
    bool Inserted =
      PassInfoMap.insert(std::make_pair(PI.getTypeInfo(),&PI)).second;
    assert(Inserted && "Pass registered multiple times!"); Inserted=Inserted;
  }
  
  void UnregisterPass(const PassInfo &PI) {
    MapType::iterator I = PassInfoMap.find(PI.getTypeInfo());
    assert(I != PassInfoMap.end() && "Pass registered but not in map!");
    
    // Remove pass from the map.
    PassInfoMap.erase(I);
  }
  
  void EnumerateWith(PassRegistrationListener *L) {
    for (MapType::const_iterator I = PassInfoMap.begin(),
         E = PassInfoMap.end(); I != E; ++I)
      L->passEnumerate(I->second);
  }
  
  
  /// Analysis Group Mechanisms.
  void RegisterAnalysisGroup(PassInfo *InterfaceInfo,
                             const PassInfo *ImplementationInfo,
                             bool isDefault) {
    AnalysisGroupInfo &AGI = AnalysisGroupInfoMap[InterfaceInfo];
    assert(AGI.Implementations.count(ImplementationInfo) == 0 &&
           "Cannot add a pass to the same analysis group more than once!");
    AGI.Implementations.insert(ImplementationInfo);
    if (isDefault) {
      assert(AGI.DefaultImpl == 0 && InterfaceInfo->getNormalCtor() == 0 &&
             "Default implementation for analysis group already specified!");
      assert(ImplementationInfo->getNormalCtor() &&
           "Cannot specify pass as default if it does not have a default ctor");
      AGI.DefaultImpl = ImplementationInfo;
      InterfaceInfo->setNormalCtor(ImplementationInfo->getNormalCtor());
    }
  }
};
}

static std::vector<PassRegistrationListener*> *Listeners = 0;

// FIXME: This should use ManagedStatic to manage the pass registrar.
// Unfortunately, we can't do this, because passes are registered with static
// ctors, and having llvm_shutdown clear this map prevents successful
// ressurection after llvm_shutdown is run.
static PassRegistrar *getPassRegistrar() {
  static PassRegistrar *PassRegistrarObj = 0;
  if (!PassRegistrarObj)
    PassRegistrarObj = new PassRegistrar();
  return PassRegistrarObj;
}

// getPassInfo - Return the PassInfo data structure that corresponds to this
// pass...
const PassInfo *Pass::getPassInfo() const {
  return lookupPassInfo(PassID);
}

const PassInfo *Pass::lookupPassInfo(intptr_t TI) {
  return getPassRegistrar()->GetPassInfo(TI);
}

void PassInfo::registerPass() {
  getPassRegistrar()->RegisterPass(*this);

  // Notify any listeners.
  if (Listeners)
    for (std::vector<PassRegistrationListener*>::iterator
           I = Listeners->begin(), E = Listeners->end(); I != E; ++I)
      (*I)->passRegistered(this);
}

void PassInfo::unregisterPass() {
  getPassRegistrar()->UnregisterPass(*this);
}

//===----------------------------------------------------------------------===//
//                  Analysis Group Implementation Code
//===----------------------------------------------------------------------===//

// RegisterAGBase implementation
//
RegisterAGBase::RegisterAGBase(const char *Name, intptr_t InterfaceID,
                               intptr_t PassID, bool isDefault)
  : PassInfo(Name, InterfaceID),
    ImplementationInfo(0), isDefaultImplementation(isDefault) {

  InterfaceInfo = const_cast<PassInfo*>(Pass::lookupPassInfo(InterfaceID));
  if (InterfaceInfo == 0) {
    // First reference to Interface, register it now.
    registerPass();
    InterfaceInfo = this;
  }
  assert(isAnalysisGroup() &&
         "Trying to join an analysis group that is a normal pass!");

  if (PassID) {
    ImplementationInfo = Pass::lookupPassInfo(PassID);
    assert(ImplementationInfo &&
           "Must register pass before adding to AnalysisGroup!");

    // Make sure we keep track of the fact that the implementation implements
    // the interface.
    PassInfo *IIPI = const_cast<PassInfo*>(ImplementationInfo);
    IIPI->addInterfaceImplemented(InterfaceInfo);
    
    getPassRegistrar()->RegisterAnalysisGroup(InterfaceInfo, IIPI, isDefault);
  }
}


//===----------------------------------------------------------------------===//
// PassRegistrationListener implementation
//

// PassRegistrationListener ctor - Add the current object to the list of
// PassRegistrationListeners...
PassRegistrationListener::PassRegistrationListener() {
  if (!Listeners) Listeners = new std::vector<PassRegistrationListener*>();
  Listeners->push_back(this);
}

// dtor - Remove object from list of listeners...
PassRegistrationListener::~PassRegistrationListener() {
  std::vector<PassRegistrationListener*>::iterator I =
    std::find(Listeners->begin(), Listeners->end(), this);
  assert(Listeners && I != Listeners->end() &&
         "PassRegistrationListener not registered!");
  Listeners->erase(I);

  if (Listeners->empty()) {
    delete Listeners;
    Listeners = 0;
  }
}

// enumeratePasses - Iterate over the registered passes, calling the
// passEnumerate callback on each PassInfo object.
//
void PassRegistrationListener::enumeratePasses() {
  getPassRegistrar()->EnumerateWith(this);
}

//===----------------------------------------------------------------------===//
//   AnalysisUsage Class Implementation
//

namespace {
  struct GetCFGOnlyPasses : public PassRegistrationListener {
    typedef AnalysisUsage::VectorType VectorType;
    VectorType &CFGOnlyList;
    GetCFGOnlyPasses(VectorType &L) : CFGOnlyList(L) {}
    
    void passEnumerate(const PassInfo *P) {
      if (P->isCFGOnlyPass())
        CFGOnlyList.push_back(P);
    }
  };
}

// setPreservesCFG - This function should be called to by the pass, iff they do
// not:
//
//  1. Add or remove basic blocks from the function
//  2. Modify terminator instructions in any way.
//
// This function annotates the AnalysisUsage info object to say that analyses
// that only depend on the CFG are preserved by this pass.
//
void AnalysisUsage::setPreservesCFG() {
  // Since this transformation doesn't modify the CFG, it preserves all analyses
  // that only depend on the CFG (like dominators, loop info, etc...)
  GetCFGOnlyPasses(Preserved).enumeratePasses();
}


