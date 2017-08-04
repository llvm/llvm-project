//===- NativeBuiltinSymbol.h -------------------------------------- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_NATIVE_NATIVEBUILTINSYMBOL_H
#define LLVM_DEBUGINFO_PDB_NATIVE_NATIVEBUILTINSYMBOL_H

#include "llvm/DebugInfo/PDB/Native/NativeRawSymbol.h"

#include "llvm/DebugInfo/PDB/PDBTypes.h"

namespace llvm {
namespace pdb {

class NativeSession;

class NativeBuiltinSymbol : public NativeRawSymbol {
public:
  NativeBuiltinSymbol(NativeSession &PDBSession, SymIndexId Id,
                      PDB_BuiltinType T, uint64_t L);
  ~NativeBuiltinSymbol() override;

  virtual std::unique_ptr<NativeRawSymbol> clone() const override;

  void dump(raw_ostream &OS, int Indent) const override;

  PDB_SymType getSymTag() const override;

  PDB_BuiltinType getBuiltinType() const override;
  bool isConstType() const override;
  uint64_t getLength() const override;
  bool isUnalignedType() const override;
  bool isVolatileType() const override;

protected:
  NativeSession &Session;
  PDB_BuiltinType Type;
  uint64_t Length;
};

} // namespace pdb
} // namespace llvm

#endif
