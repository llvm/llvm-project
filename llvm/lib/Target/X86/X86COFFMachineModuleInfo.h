//===-- X86coffmachinemoduleinfo.h - X86 COFF MMI Impl ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is an MMI implementation for X86 COFF (windows) targets.
//
//===----------------------------------------------------------------------===//

#ifndef X86COFF_MACHINEMODULEINFO_H
#define X86COFF_MACHINEMODULEINFO_H

#include "X86MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/ADT/DenseSet.h"

namespace llvm {
  class X86MachineFunctionInfo;
  class DataLayout;

/// X86COFFMachineModuleInfo - This is a MachineModuleInfoImpl implementation
/// for X86 COFF targets.
class X86COFFMachineModuleInfo : public MachineModuleInfoImpl {
  DenseSet<MCSymbol const *> Externals;
public:
  X86COFFMachineModuleInfo(const MachineModuleInfo &) {}
  virtual ~X86COFFMachineModuleInfo();

  void addExternalFunction(MCSymbol* Symbol) {
    Externals.insert(Symbol);
  }

  typedef DenseSet<MCSymbol const *>::const_iterator externals_iterator;
  externals_iterator externals_begin() const { return Externals.begin(); }
  externals_iterator externals_end() const { return Externals.end(); }
};



} // end namespace llvm

#endif
