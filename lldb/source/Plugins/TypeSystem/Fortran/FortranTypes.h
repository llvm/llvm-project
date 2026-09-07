//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the classes that describe the Fortran types.
///
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H

#include "Plugins/SymbolFile/DWARF/DWARFDIE.h"
#include "lldb/Expression/DWARFExpressionList.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/ConstString.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"

namespace lldb_private {
namespace plugin {
namespace fortran {

/// A simplified internal representation of a Fortran type.
class FortranType : public llvm::FoldingSetNode {
  // LLVM RTTI support
  static char ID;

public:
  // llvm casting support
  virtual bool isA(const void *ClassID) const { return ClassID == &ID; }

  static bool classof(const FortranType *ft) { return ft->isA(&ID); }

  enum TypeKind {
    KIND_INTEGER,
    KIND_LOGICAL,
    KIND_REAL,
    KIND_COMPLEX,
    KIND_UNKNOWN
  };

  FortranType(int32_t kind, uint64_t bitsize, const ConstString &name)
      : m_kind(kind), m_bitsize(bitsize), m_type_name(name) {}
  virtual ~FortranType();
  int GetKind() const { return m_kind; }
  uint64_t GetBitSize() const { return m_bitsize; }
  ConstString GetName() const { return m_type_name; }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, m_kind, m_bitsize);
  }

  static void Profile(llvm::FoldingSetNodeID &ID, int32_t kind,
                      uint64_t bitsize) {
    ID.AddInteger(kind);
    ID.AddInteger(bitsize);
  }

private:
  int32_t m_kind;
  uint64_t m_bitsize;
  ConstString m_type_name;
};

} // namespace fortran
} // namespace plugin
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H