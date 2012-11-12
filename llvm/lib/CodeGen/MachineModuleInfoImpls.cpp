//===-- llvm/CodeGen/MachineModuleInfoImpls.cpp ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/MC/MCSymbol.h"
using namespace llvm;

//===----------------------------------------------------------------------===//
// MachineModuleInfoMachO
//===----------------------------------------------------------------------===//

// Out of line virtual method.
void MachineModuleInfoMachO::anchor() {}
void MachineModuleInfoELF::anchor() {}

static int SortSymbolPair(const void *LHS, const void *RHS) {
  typedef std::pair<MCSymbol*, MachineModuleInfoImpl::StubValueTy> PairTy;
  const MCSymbol *LHSS = ((const PairTy *)LHS)->first;
  const MCSymbol *RHSS = ((const PairTy *)RHS)->first;
  return LHSS->getName().compare(RHSS->getName());
}

/// GetSortedStubs - Return the entries from a DenseMap in a deterministic
/// sorted orer.
MachineModuleInfoImpl::SymbolListTy
MachineModuleInfoImpl::GetSortedStubs(const DenseMap<MCSymbol*,
                                      MachineModuleInfoImpl::StubValueTy>&Map) {
  MachineModuleInfoImpl::SymbolListTy List(Map.begin(), Map.end());

  if (!List.empty())
    qsort(&List[0], List.size(), sizeof(List[0]), SortSymbolPair);
  return List;
}

