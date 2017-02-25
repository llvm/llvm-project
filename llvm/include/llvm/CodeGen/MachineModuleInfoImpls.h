//===-- llvm/CodeGen/MachineModuleInfoImpls.h -------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H
#define LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H

#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/CodeGen/MachineModuleInfo.h"

namespace llvm {
class MCSymbol;

/// MachineModuleInfoMachO - This is a MachineModuleInfoImpl implementation
/// for MachO targets.
class MachineModuleInfoMachO : public MachineModuleInfoImpl {
  /// GVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something like
  /// "Lfoo$non_lazy_ptr", the value is something like "_foo". The extra bit
  /// is true if this GV is external.
  DenseMap<MCSymbol *, StubValueTy> GVStubs;

  /// ThreadLocalGVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something
  /// like "Lfoo$non_lazy_ptr", the value is something like "_foo". The extra
  /// bit is true if this GV is external.
  DenseMap<MCSymbol *, StubValueTy> ThreadLocalGVStubs;

  virtual void anchor(); // Out of line virtual method.
public:
  MachineModuleInfoMachO(const MachineModuleInfo &) {}

  StubValueTy &getGVStubEntry(MCSymbol *Sym) {
    assert(Sym && "Key cannot be null");
    return GVStubs[Sym];
  }

  StubValueTy &getThreadLocalGVStubEntry(MCSymbol *Sym) {
    assert(Sym && "Key cannot be null");
    return ThreadLocalGVStubs[Sym];
  }

  /// Accessor methods to return the set of stubs in sorted order.
  SymbolListTy GetGVStubList() { return getSortedStubs(GVStubs); }
  SymbolListTy GetThreadLocalGVStubList() {
    return getSortedStubs(ThreadLocalGVStubs);
  }
};

/// MachineModuleInfoELF - This is a MachineModuleInfoImpl implementation
/// for ELF targets.
class MachineModuleInfoELF : public MachineModuleInfoImpl {
  /// GVStubs - These stubs are used to materialize global addresses in PIC
  /// mode.
  DenseMap<MCSymbol *, StubValueTy> GVStubs;

  virtual void anchor(); // Out of line virtual method.
public:
  MachineModuleInfoELF(const MachineModuleInfo &) {}

  StubValueTy &getGVStubEntry(MCSymbol *Sym) {
    assert(Sym && "Key cannot be null");
    return GVStubs[Sym];
  }

  /// Accessor methods to return the set of stubs in sorted order.

  SymbolListTy GetGVStubList() { return getSortedStubs(GVStubs); }
};

/// MachineModuleInfoWasm - This is a MachineModuleInfoImpl implementation
/// for Wasm targets.
class MachineModuleInfoWasm : public MachineModuleInfoImpl {
  /// GVStubs - These stubs are used to materialize global addresses in PIC
  /// mode.
  std::vector<MVT> Globals;
  unsigned StackPointerGlobal;

  virtual void anchor(); // Out of line virtual method.
public:
  MachineModuleInfoWasm(const MachineModuleInfo &)
    : StackPointerGlobal(-1U) {}

  void addGlobal(MVT VT) { Globals.push_back(VT); }
  const std::vector<MVT> &getGlobals() const { return Globals; }

  bool hasStackPointerGlobal() const {
    return StackPointerGlobal != -1U;
  }
  unsigned getStackPointerGlobal() const {
    assert(hasStackPointerGlobal() && "Stack ptr global hasn't been set");
    return StackPointerGlobal;
  }
  void setStackPointerGlobal(unsigned Global) { StackPointerGlobal = Global; }
};

} // end namespace llvm

#endif
