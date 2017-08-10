//===- NativeEnumSymbol.h - info about enum type ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NATIVEENUMSYMBOL_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NATIVEENUMSYMBOL_H

#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/CodeView/TypeVisitorCallbacks.h"
#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeSession.h"

namespace llvm {
namespace pdb {

class NativeEnumSymbol : public NativeRawSymbol,
                         public codeview::TypeVisitorCallbacks {
public:
  NativeEnumSymbol(NativeSession &Session, SymIndexId Id,
                   const codeview::CVType &CV);
  ~NativeEnumSymbol() override;

  std::unique_ptr<NativeRawSymbol> clone() const override;

  std::unique_ptr<IPDBEnumSymbols>
  findChildren(PDB_SymType Type) const override;

  Error visitKnownRecord(codeview::CVType &CVR,
                         codeview::EnumRecord &Record) override;
  Error visitKnownMember(codeview::CVMemberRecord &CVM,
                         codeview::EnumeratorRecord &Record) override;

  PDB_SymType getSymTag() const override;
  uint32_t getClassParentId() const override;
  uint32_t getUnmodifiedTypeId() const override;
  bool hasConstructor() const override;
  bool hasAssignmentOperator() const override;
  bool hasCastOperator() const override;
  uint64_t getLength() const override;
  std::string getName() const override;
  bool isNested() const override;
  bool hasOverloadedOperator() const override;
  bool isPacked() const override;
  bool isScoped() const override;
  uint32_t getTypeId() const override;

protected:
  codeview::CVType CV;
  codeview::EnumRecord Record;
};

} // namespace pdb
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_NATIVEENUMSYMBOL_H
