//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_UTILS_TABLEGEN_COMMON_RUNTIMELIBCALLS_H
#define LLVM_UTILS_TABLEGEN_COMMON_RUNTIMELIBCALLS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/SetTheory.h"

namespace llvm {

class AvailabilityPredicate {
  const Record *TheDef;
  StringRef PredicateString;

public:
  AvailabilityPredicate(const Record *Def) : TheDef(Def) {
    if (TheDef)
      PredicateString = TheDef->getValueAsString("Cond");
  }

  const Record *getDef() const { return TheDef; }

  bool isAlwaysAvailable() const { return PredicateString.empty(); }

  void emitIf(raw_ostream &OS) const {
    OS << "if (" << PredicateString << ") {\n";
  }

  void emitEndIf(raw_ostream &OS) const { OS << "}\n"; }

  void emitTableVariableNameSuffix(raw_ostream &OS) const {
    if (TheDef)
      OS << '_' << TheDef->getName();
  }
};

class RuntimeLibcalls;
class RuntimeLibcallImpl;

/// Used to apply predicates to nested sets of libcalls.
struct LibcallPredicateExpander : SetTheory::Expander {
  const RuntimeLibcalls &Libcalls;
  DenseMap<const RuntimeLibcallImpl *,
           std::pair<std::vector<const Record *>, const Record *>> &Func2Preds;

  LibcallPredicateExpander(
      const RuntimeLibcalls &Libcalls,
      DenseMap<const RuntimeLibcallImpl *,
               std::pair<std::vector<const Record *>, const Record *>>
          &Func2Preds)
      : Libcalls(Libcalls), Func2Preds(Func2Preds) {}

  void expand(SetTheory &ST, const Record *Def,
              SetTheory::RecSet &Elts) override;
};

class RuntimeLibcall {
  const Record *TheDef = nullptr;
  const size_t EnumVal;

public:
  RuntimeLibcall() = delete;
  RuntimeLibcall(const Record *Def, size_t EnumVal)
      : TheDef(Def), EnumVal(EnumVal) {
    assert(Def);
  }

  ~RuntimeLibcall() { assert(TheDef); }

  const Record *getDef() const { return TheDef; }

  StringRef getName() const { return TheDef->getName(); }

  size_t getEnumVal() const { return EnumVal; }

  void emitEnumEntry(raw_ostream &OS) const {
    OS << "RTLIB::" << TheDef->getValueAsString("Name");
  }
};

class RuntimeLibcallImpl {
  const Record *TheDef;
  const RuntimeLibcall *Provides = nullptr;
  const size_t EnumVal;

public:
  RuntimeLibcallImpl(
      const Record *Def,
      const DenseMap<const Record *, const RuntimeLibcall *> &ProvideMap,
      size_t EnumVal)
      : TheDef(Def), EnumVal(EnumVal) {
    if (const Record *ProvidesDef = Def->getValueAsDef("Provides"))
      Provides = ProvideMap.lookup(ProvidesDef);
  }

  ~RuntimeLibcallImpl() = default;

  const Record *getDef() const { return TheDef; }

  StringRef getName() const { return TheDef->getName(); }

  size_t getEnumVal() const { return EnumVal; }

  const RuntimeLibcall *getProvides() const { return Provides; }

  StringRef getLibcallFuncName() const {
    return TheDef->getValueAsString("LibCallFuncName");
  }

  const Record *getCallingConv() const {
    return TheDef->getValueAsOptionalDef("CallingConv");
  }

  void emitQuotedLibcallFuncName(raw_ostream &OS) const {
    OS << '\"' << getLibcallFuncName() << '\"';
  }

  bool isDefault() const { return TheDef->getValueAsBit("IsDefault"); }

  void emitEnumEntry(raw_ostream &OS) const {
    OS << "RTLIB::impl_" << this->getName();
  }

  void emitSetImplCall(raw_ostream &OS) const {
    OS << "setLibcallImpl(";
    Provides->emitEnumEntry(OS);
    OS << ", ";
    emitEnumEntry(OS);
    OS << "); // " << getLibcallFuncName() << '\n';
  }

  void emitTableEntry(raw_ostream &OS) const {
    OS << '{';
    Provides->emitEnumEntry(OS);
    OS << ", ";
    emitEnumEntry(OS);
    OS << "}, // " << getLibcallFuncName() << '\n';
  }

  void emitSetCallingConv(raw_ostream &OS) const {}
};

struct LibcallsWithCC {
  std::vector<const RuntimeLibcallImpl *> LibcallImpls;
  const Record *CallingConv = nullptr;
};

class RuntimeLibcalls {
private:
  DenseMap<const Record *, const RuntimeLibcall *> Def2RuntimeLibcall;
  DenseMap<const Record *, const RuntimeLibcallImpl *> Def2RuntimeLibcallImpl;

  std::vector<RuntimeLibcall> RuntimeLibcallDefList;
  std::vector<RuntimeLibcallImpl> RuntimeLibcallImplDefList;

  DenseMap<const RuntimeLibcall *, const RuntimeLibcallImpl *>
      LibCallToDefaultImpl;

public:
  RuntimeLibcalls(const RecordKeeper &Records);

  ArrayRef<RuntimeLibcall> getRuntimeLibcallDefList() const {
    return RuntimeLibcallDefList;
  }

  ArrayRef<RuntimeLibcallImpl> getRuntimeLibcallImplDefList() const {
    return RuntimeLibcallImplDefList;
  }

  const RuntimeLibcall *getRuntimeLibcall(const Record *Def) const {
    return Def2RuntimeLibcall.lookup(Def);
  }

  const RuntimeLibcallImpl *getRuntimeLibcallImpl(const Record *Def) const {
    return Def2RuntimeLibcallImpl.lookup(Def);
  }
};

} // namespace llvm

#endif // LLVM_UTILS_TABLEGEN_COMMON_RUNTIMELIBCALLS_H
