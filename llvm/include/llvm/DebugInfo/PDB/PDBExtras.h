//===- PDBExtras.h - helper functions and classes for PDBs ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_PDBEXTRAS_H
#define LLVM_DEBUGINFO_PDB_PDBEXTRAS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/DebugInfo/CodeView/CodeView.h"
#include "llvm/DebugInfo/PDB/PDBTypes.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <unordered_map>

namespace llvm {

namespace pdb {

using TagStats = std::unordered_map<PDB_SymType, int>;

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_VariantType &Value);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_CallingConv &Conv);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_BuiltinType &Type);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_DataKind &Data);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const llvm::codeview::CPURegister &CpuReg);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_LocType &Loc);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const codeview::ThunkOrdinal &Thunk);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_Checksum &Checksum);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_Lang &Lang);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_SymType &Tag);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS,
                                 const PDB_MemberAccess &Access);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_UdtType &Type);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const PDB_Machine &Machine);

LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const Variant &Value);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const VersionInfo &Version);
LLVM_ABI raw_ostream &operator<<(raw_ostream &OS, const TagStats &Stats);

LLVM_ABI raw_ostream &dumpPDBSourceCompression(raw_ostream &OS,
                                               uint32_t Compression);

template <typename T>
void dumpSymbolField(raw_ostream &OS, StringRef Name, T Value, int Indent) {
  OS << "\n";
  OS.indent(Indent);
  OS << Name << ": " << Value;
}

} // end namespace pdb

} // end namespace llvm

#endif // LLVM_DEBUGINFO_PDB_PDBEXTRAS_H
