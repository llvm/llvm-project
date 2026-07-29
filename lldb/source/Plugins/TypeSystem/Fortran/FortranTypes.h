//===-- FortranTypes.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H
#define LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H

#include "lldb/Expression/DWARFExpressionList.h"
#include "lldb/Symbol/CompilerType.h"
#include "lldb/Utility/ConstString.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"

namespace lldb_private {
namespace plugin {
namespace fortran {

/// A simplified internal representation of a Fortran basic type.
/// Types that need more information than this will inherit from this class.
class FortranType : public llvm::FoldingSetNode {
public:
  enum TypeKind {
    KIND_INTEGER,
    KIND_LOGICAL,
    KIND_REAL,
    KIND_COMPLEX,
    KIND_FUNCTION,
    KIND_UNKNOWN
  };
  FortranType(int32_t kind, uint64_t bitsize, const ConstString &name)
      : m_kind(kind), m_bitsize(bitsize), m_type_name(name) {}
  virtual ~FortranType() = default;
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

/// We represent Functions as types to satisfy lldb's requirement for everything
/// to be a type.
class FortranFunction : public FortranType {
public:
  FortranFunction(ConstString func_name,
                  const llvm::SmallVectorImpl<CompilerType> &parameters)
      : FortranType(FortranType::KIND_FUNCTION, 0, func_name) {
    m_parameters.assign(parameters.begin(), parameters.end());
  }
  llvm::ArrayRef<CompilerType> GetParameters() const { return m_parameters; }
  size_t GetNumberOfParameters() const { return m_parameters.size(); }

  void Profile(llvm::FoldingSetNodeID &id) const {
    Profile(id, GetName(), m_parameters);
  }

  static void Profile(llvm::FoldingSetNodeID &id, ConstString func_name,
                      llvm::ArrayRef<CompilerType> parameters) {
    id.AddString(func_name.GetStringRef());
    id.AddInteger(parameters.size());

    for (const auto &param : parameters)
      id.AddPointer(param.GetOpaqueQualType());
  }

private:
  llvm::SmallVector<CompilerType, 4> m_parameters;
};
} // namespace fortran
} // namespace plugin
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H