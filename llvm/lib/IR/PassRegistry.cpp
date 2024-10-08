//===- PassRegistry.cpp - Pass Registration Implementation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the PassRegistry, with which passes are registered on
// initialization, and supports the PassManager in dependency resolution.
//
//===----------------------------------------------------------------------===//

#include "llvm/PassRegistry.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Pass.h"
#include "llvm/PassInfo.h"
#include <cassert>
#include <memory>
#include <utility>

using namespace llvm;

PassRegistry *PassRegistry::getPassRegistry() {
  static PassRegistry PassRegistryObj;
  return &PassRegistryObj;
}

//===----------------------------------------------------------------------===//
// Accessors
//

PassRegistry::~PassRegistry() = default;

const PassInfo *PassRegistry::getPassInfo(const void *TI) const {
  sys::SmartScopedReader<true> Guard(Lock);
  return PassInfoMap.lookup(TI);
}

const PassInfo *PassRegistry::getPassInfo(StringRef Arg) const {
  sys::SmartScopedReader<true> Guard(Lock);
  return PassInfoStringMap.lookup(Arg);
}

//===----------------------------------------------------------------------===//
// Pass Registration mechanism
//

void PassRegistry::registerPass(const PassInfo &PI, bool ShouldFree) {
  sys::SmartScopedWriter<true> Guard(Lock);
  bool Inserted =
      PassInfoMap.insert(std::make_pair(PI.getTypeInfo(), &PI)).second;
  assert(Inserted && "Pass registered multiple times!");
  (void)Inserted;
  PassInfoStringMap[PI.getPassArgument()] = &PI;

  // Notify any listeners.
  for (auto *Listener : Listeners)
    Listener->passRegistered(&PI);

  if (ShouldFree)
    ToFree.push_back(std::unique_ptr<const PassInfo>(&PI));
}

void PassRegistry::enumerateWith(PassRegistrationListener *L) {
  sys::SmartScopedReader<true> Guard(Lock);
  for (auto PassInfoPair : PassInfoMap)
    L->passEnumerate(PassInfoPair.second);
}

void PassRegistry::addRegistrationListener(PassRegistrationListener *L) {
  sys::SmartScopedWriter<true> Guard(Lock);
  Listeners.push_back(L);
}

void PassRegistry::removeRegistrationListener(PassRegistrationListener *L) {
  sys::SmartScopedWriter<true> Guard(Lock);

  auto I = llvm::find(Listeners, L);
  Listeners.erase(I);
}
