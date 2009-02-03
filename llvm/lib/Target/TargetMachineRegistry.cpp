//===-- TargetMachineRegistry.cpp - Target Auto Registration Impl ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes the RegisterTarget class, which TargetMachine
// implementations should use to register themselves with the system.  This file
// also exposes the TargetMachineRegistry class, which allows tools to inspect
// all of registered targets.
//
//===----------------------------------------------------------------------===//

#include "llvm/Target/TargetMachineRegistry.h"
#include <algorithm>
using namespace llvm;

/// getClosestStaticTargetForModule - Given an LLVM module, pick the best target
/// that is compatible with the module.  If no close target can be found, this
/// returns null and sets the Error string to a reason.
const TargetMachineRegistry::entry *
TargetMachineRegistry::getClosestStaticTargetForModule(const Module &M,
                                                       std::string &Error) {
  std::vector<std::pair<unsigned, const entry *> > UsableTargets;
  for (Registry<TargetMachine>::iterator I = begin(), E = end(); I != E; ++I)
    if (unsigned Qual = I->ModuleMatchQualityFn(M))
      UsableTargets.push_back(std::make_pair(Qual, &*I));

  if (UsableTargets.empty()) {
    Error = "No available targets are compatible with this module";
    return 0;
  } else if (UsableTargets.size() == 1)
    return UsableTargets.back().second;

  // Otherwise, take the best target, but make sure we don't have two equally
  // good best targets.
  std::sort(UsableTargets.begin(), UsableTargets.end());
  if (UsableTargets.back().first ==UsableTargets[UsableTargets.size()-2].first){
    Error = "Cannot choose between targets \"" +
      std::string(UsableTargets.back().second->Name) + "\" and \"" +
      std::string(UsableTargets[UsableTargets.size()-2].second->Name) + "\"";
    return 0;
  }
  return UsableTargets.back().second;
}

/// getClosestTargetForJIT - Pick the best target that is compatible with
/// the current host.  If no close target can be found, this returns null
/// and sets the Error string to a reason.
const TargetMachineRegistry::entry *
TargetMachineRegistry::getClosestTargetForJIT(std::string &Error) {
  std::vector<std::pair<unsigned, const entry *> > UsableTargets;
  for (Registry<TargetMachine>::iterator I = begin(), E = end(); I != E; ++I)
    if (unsigned Qual = I->JITMatchQualityFn())
      UsableTargets.push_back(std::make_pair(Qual, &*I));

  if (UsableTargets.empty()) {
    Error = "No JIT is available for this host";
    return 0;
  } else if (UsableTargets.size() == 1)
    return UsableTargets.back().second;

  // Otherwise, take the best target.  If there is a tie, just pick one.
  unsigned MaxQual = UsableTargets.front().first;
  const entry *MaxQualTarget = UsableTargets.front().second;

  for (unsigned i = 1, e = UsableTargets.size(); i != e; ++i)
    if (UsableTargets[i].first > MaxQual) {
      MaxQual = UsableTargets[i].first;
      MaxQualTarget = UsableTargets[i].second;
    }

  return MaxQualTarget;
}

