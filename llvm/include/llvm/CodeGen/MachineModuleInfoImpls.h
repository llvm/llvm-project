//===- llvm/CodeGen/MachineModuleInfoImpls.h --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines object-file format specific implementations of
// MachineModuleInfoImpl.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H
#define LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include <cassert>

namespace llvm {

class MCSymbol;

/// MachineModuleInfoMachO - This is a MachineModuleInfoImpl implementation
/// for MachO targets.
class MachineModuleInfoMachO : public MachineModuleInfoImpl {
public:
  /// The information specific to a Darwin '$auth_ptr' stub.
  struct AuthStubInfo {
    const MCExpr *Pointer;
  };

private:
  /// GVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something like
  /// "Lfoo$non_lazy_ptr", the value is something like "_foo". The extra bit
  /// is true if this GV is external.
  DenseMap<MCSymbol *, StubValueTy> GVStubs;

  /// ThreadLocalGVStubs - Darwin '$non_lazy_ptr' stubs.  The key is something
  /// like "Lfoo$non_lazy_ptr", the value is something like "_foo". The extra
  /// bit is true if this GV is external.
  DenseMap<MCSymbol *, StubValueTy> ThreadLocalGVStubs;

  /// Darwin '$auth_ptr' stubs.  The key is the stub symbol, like
  /// "Lfoo$addend$auth_ptr$ib$12".  The value is the MCExpr representing that
  /// pointer, something like "_foo+addend@AUTH(ib, 12)".
  DenseMap<MCSymbol *, AuthStubInfo> AuthGVStubs;

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

  AuthStubInfo &getAuthGVStubEntry(MCSymbol *Sym) {
    assert(Sym && "Key cannot be null");
    return AuthGVStubs[Sym];
  }

  /// Accessor methods to return the set of stubs in sorted order.
  SymbolListTy GetGVStubList() { return getSortedStubs(GVStubs); }
  SymbolListTy GetThreadLocalGVStubList() {
    return getSortedStubs(ThreadLocalGVStubs);
  }

  typedef std::pair<MCSymbol *, AuthStubInfo> AuthStubPairTy;
  typedef std::vector<AuthStubPairTy> AuthStubListTy;

  AuthStubListTy getAuthGVStubList() {
    AuthStubListTy List(AuthGVStubs.begin(), AuthGVStubs.end());

    if (!List.empty())
      std::sort(List.begin(), List.end(),
                [](const AuthStubPairTy &LHS, const AuthStubPairTy &RHS) {
                  return LHS.first->getName() < RHS.first->getName();
                });

    AuthGVStubs.clear();
    return List;
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

/// MachineModuleInfoCOFF - This is a MachineModuleInfoImpl implementation
/// for COFF targets.
class MachineModuleInfoCOFF : public MachineModuleInfoImpl {
  /// GVStubs - These stubs are used to materialize global addresses in PIC
  /// mode.
  DenseMap<MCSymbol *, StubValueTy> GVStubs;

  virtual void anchor(); // Out of line virtual method.

public:
  MachineModuleInfoCOFF(const MachineModuleInfo &) {}

  StubValueTy &getGVStubEntry(MCSymbol *Sym) {
    assert(Sym && "Key cannot be null");
    return GVStubs[Sym];
  }

  /// Accessor methods to return the set of stubs in sorted order.

  SymbolListTy GetGVStubList() { return getSortedStubs(GVStubs); }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MACHINEMODULEINFOIMPLS_H
