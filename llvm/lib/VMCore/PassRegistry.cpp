//===- PassRegistry.cpp - Pass Registration Implementation ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PassRegistry, with which passes are registered on
// initialization, and supports the PassManager in dependency resolution.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassRegistry.h"
#include "llvm/PassSupport.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Function.h"
#include <vector>

using namespace llvm;

// FIXME: We use ManagedStatic to erase the pass registrar on shutdown.
// Unfortunately, passes are registered with static ctors, and having
// llvm_shutdown clear this map prevents successful resurrection after
// llvm_shutdown is run.  Ideally we should find a solution so that we don't
// leak the map, AND can still resurrect after shutdown.
static ManagedStatic<PassRegistry> PassRegistryObj;
PassRegistry *PassRegistry::getPassRegistry() {
  return &*PassRegistryObj;
}

static ManagedStatic<sys::SmartMutex<true> > Lock;

//===----------------------------------------------------------------------===//
// PassRegistryImpl
//

namespace {
struct PassRegistryImpl {
  /// PassInfoMap - Keep track of the PassInfo object for each registered pass.
  typedef DenseMap<const void*, const PassInfo*> MapType;
  MapType PassInfoMap;
  
  typedef StringMap<const PassInfo*> StringMapType;
  StringMapType PassInfoStringMap;
  
  /// AnalysisGroupInfo - Keep track of information for each analysis group.
  struct AnalysisGroupInfo {
    SmallPtrSet<const PassInfo *, 8> Implementations;
  };
  DenseMap<const PassInfo*, AnalysisGroupInfo> AnalysisGroupInfoMap;
  
  std::vector<const PassInfo*> ToFree;
  std::vector<PassRegistrationListener*> Listeners;
};
} // end anonymous namespace

void *PassRegistry::getImpl() const {
  if (!pImpl)
    pImpl = new PassRegistryImpl();
  return pImpl;
}

//===----------------------------------------------------------------------===//
// Accessors
//

PassRegistry::~PassRegistry() {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(pImpl);
  
  for (std::vector<const PassInfo*>::iterator I = Impl->ToFree.begin(),
       E = Impl->ToFree.end(); I != E; ++I)
    delete *I;
  
  delete Impl;
  pImpl = 0;
}

const PassInfo *PassRegistry::getPassInfo(const void *TI) const {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  PassRegistryImpl::MapType::const_iterator I = Impl->PassInfoMap.find(TI);
  return I != Impl->PassInfoMap.end() ? I->second : 0;
}

const PassInfo *PassRegistry::getPassInfo(StringRef Arg) const {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  PassRegistryImpl::StringMapType::const_iterator
    I = Impl->PassInfoStringMap.find(Arg);
  return I != Impl->PassInfoStringMap.end() ? I->second : 0;
}

//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//

void PassRegistry::registerPass(const PassInfo &PI, bool ShouldFree) {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  bool Inserted =
    Impl->PassInfoMap.insert(std::make_pair(PI.getTypeInfo(),&PI)).second;
  assert(Inserted && "Pass registered multiple times!");
  (void)Inserted;
  Impl->PassInfoStringMap[PI.getPassArgument()] = &PI;
  
  // Notify any listeners.
  for (std::vector<PassRegistrationListener*>::iterator
       I = Impl->Listeners.begin(), E = Impl->Listeners.end(); I != E; ++I)
    (*I)->passRegistered(&PI);
  
  if (ShouldFree) Impl->ToFree.push_back(&PI);
}

void PassRegistry::unregisterPass(const PassInfo &PI) {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  PassRegistryImpl::MapType::iterator I = 
    Impl->PassInfoMap.find(PI.getTypeInfo());
  assert(I != Impl->PassInfoMap.end() && "Pass registered but not in map!");
  
  // Remove pass from the map.
  Impl->PassInfoMap.erase(I);
  Impl->PassInfoStringMap.erase(PI.getPassArgument());
}

void PassRegistry::enumerateWith(PassRegistrationListener *L) {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  for (PassRegistryImpl::MapType::const_iterator I = Impl->PassInfoMap.begin(),
       E = Impl->PassInfoMap.end(); I != E; ++I)
    L->passEnumerate(I->second);
}


/// Analysis Group Mechanisms.
void PassRegistry::registerAnalysisGroup(const void *InterfaceID, 
                                         const void *PassID,
                                         PassInfo& Registeree,
                                         bool isDefault,
                                         bool ShouldFree) {
  PassInfo *InterfaceInfo =  const_cast<PassInfo*>(getPassInfo(InterfaceID));
  if (InterfaceInfo == 0) {
    // First reference to Interface, register it now.
    registerPass(Registeree);
    InterfaceInfo = &Registeree;
  }
  assert(Registeree.isAnalysisGroup() && 
         "Trying to join an analysis group that is a normal pass!");

  if (PassID) {
    PassInfo *ImplementationInfo = const_cast<PassInfo*>(getPassInfo(PassID));
    assert(ImplementationInfo &&
           "Must register pass before adding to AnalysisGroup!");

    sys::SmartScopedLock<true> Guard(*Lock);
    
    // Make sure we keep track of the fact that the implementation implements
    // the interface.
    ImplementationInfo->addInterfaceImplemented(InterfaceInfo);

    PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
    PassRegistryImpl::AnalysisGroupInfo &AGI =
      Impl->AnalysisGroupInfoMap[InterfaceInfo];
    assert(AGI.Implementations.count(ImplementationInfo) == 0 &&
           "Cannot add a pass to the same analysis group more than once!");
    AGI.Implementations.insert(ImplementationInfo);
    if (isDefault) {
      assert(InterfaceInfo->getNormalCtor() == 0 &&
             "Default implementation for analysis group already specified!");
      assert(ImplementationInfo->getNormalCtor() &&
           "Cannot specify pass as default if it does not have a default ctor");
      InterfaceInfo->setNormalCtor(ImplementationInfo->getNormalCtor());
    }
  }
  
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  if (ShouldFree) Impl->ToFree.push_back(&Registeree);
}

void PassRegistry::addRegistrationListener(PassRegistrationListener *L) {
  sys::SmartScopedLock<true> Guard(*Lock);
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  Impl->Listeners.push_back(L);
}

void PassRegistry::removeRegistrationListener(PassRegistrationListener *L) {
  sys::SmartScopedLock<true> Guard(*Lock);
  
  // NOTE: This is necessary, because removeRegistrationListener() can be called
  // as part of the llvm_shutdown sequence.  Since we have no control over the
  // order of that sequence, we need to gracefully handle the case where the
  // PassRegistry is destructed before the object that triggers this call.
  if (!pImpl) return;
  
  PassRegistryImpl *Impl = static_cast<PassRegistryImpl*>(getImpl());
  std::vector<PassRegistrationListener*>::iterator I =
    std::find(Impl->Listeners.begin(), Impl->Listeners.end(), L);
  assert(I != Impl->Listeners.end() &&
         "PassRegistrationListener not registered!");
  Impl->Listeners.erase(I);
}
