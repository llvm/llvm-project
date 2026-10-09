//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NATIVETYPETYPEDEFALIAS_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NATIVETYPETYPEDEFALIAS_H

#include "llvm/DebugInfo/CodeView/TypeRecord.h"
#include "llvm/DebugInfo/PDB/IPDBRawSymbol.h"
#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"

namespace llvm {

class raw_ostream;

namespace pdb {

class NativeSession;

/// A typedef from the TPI stream (LF_ALIAS).
class LLVM_ABI NativeTypeTypedefAlias : public NativeRawSymbol {
public:
  NativeTypeTypedefAlias(NativeSession &Session, SymIndexId Id,
                         codeview::TypeIndex TI, codeview::AliasRecord Typedef);

  NativeTypeTypedefAlias(NativeSession &Session, SymIndexId Id,
                         NativeTypeTypedefAlias &UnmodifiedType,
                         codeview::ModifierRecord Modifier);

  ~NativeTypeTypedefAlias() override;

  void dump(raw_ostream &OS, int Indent, PdbSymbolIdField ShowIdFields,
            PdbSymbolIdField RecurseIdFields) const override;

  std::string getName() const override;
  SymIndexId getTypeId() const override;

  SymIndexId getUnmodifiedTypeId() const override;
  bool isConstType() const override;
  bool isUnalignedType() const override;
  bool isVolatileType() const override;

protected:
  codeview::AliasRecord Record;
  NativeTypeTypedefAlias *UnmodifiedType = nullptr;
  std::optional<codeview::ModifierRecord> Modifiers;
};

} // namespace pdb
} // namespace llvm

#endif // LLVM_DEBUGINFO_PDB_NATIVE_NATIVETYPETYPEDEFALIAS_H
