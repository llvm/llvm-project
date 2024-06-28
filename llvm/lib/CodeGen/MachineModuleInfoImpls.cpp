//===- llvm/CodeGen/MachineModuleInfoImpls.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/MC/MCSymbol.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineModuleInfoMachO
//===----------------------------------------------------------------------===//

// Out of line virtual method.
void MachineModuleInfoMachO::anchor() {}
void MachineModuleInfoELF::anchor() {}
void MachineModuleInfoCOFF::anchor() {}
void MachineModuleInfoWasm::anchor() {}

using PairTy = std::pair<MCSymbol *, MachineModuleInfoImpl::StubValueTy>;
static int SortSymbolPair(const PairTy *LHS, const PairTy *RHS) {
  return LHS->first->getName().compare(RHS->first->getName());
}

MachineModuleInfoImpl::SymbolListTy MachineModuleInfoImpl::getSortedStubs(
    DenseMap<MCSymbol *, MachineModuleInfoImpl::StubValueTy> &Map) {
  MachineModuleInfoImpl::SymbolListTy List(Map.begin(), Map.end());

  array_pod_sort(List.begin(), List.end(), SortSymbolPair);

  Map.clear();
  return List;
}

template <typename MachineModuleInfoTarget>
static typename MachineModuleInfoTarget::AuthStubListTy getAuthGVStubListHelper(
    DenseMap<MCSymbol *, typename MachineModuleInfoTarget::AuthStubInfo>
        &AuthPtrStubs) {
  typename MachineModuleInfoTarget::AuthStubListTy List(AuthPtrStubs.begin(),
                                                        AuthPtrStubs.end());

  if (!List.empty())
    llvm::sort(List.begin(), List.end(),
               [](const typename MachineModuleInfoTarget::AuthStubPairTy &LHS,
                  const typename MachineModuleInfoTarget::AuthStubPairTy &RHS) {
                 return LHS.first->getName() < RHS.first->getName();
               });

  AuthPtrStubs.clear();
  return List;
}

MachineModuleInfoELF::AuthStubListTy MachineModuleInfoELF::getAuthGVStubList() {
  return getAuthGVStubListHelper<MachineModuleInfoELF>(AuthPtrStubs);
}
