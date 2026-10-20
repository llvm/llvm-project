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
    KIND_UNSIGNED,
    KIND_LOGICAL,
    KIND_REAL,
    KIND_COMPLEX,
    KIND_FUNCTION,
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

class FortranFunction : public FortranType {
public:
  FortranFunction(ConstString func_name,
                  const llvm::SmallVectorImpl<CompilerType> &parameters,
                  const llvm::SmallVectorImpl<llvm::StringRef> &parameter_names,
                  CompilerType return_type);

  llvm::ArrayRef<CompilerType> GetParameters() const { return m_parameters; }
  size_t GetNumberOfParameters() const { return m_parameters.size(); }
  CompilerType GetReturnType() const { return m_return_type; }
  ConstString GetFunctionSignature() const { return m_function_signature; };

  void Profile(llvm::FoldingSetNodeID &id) const {
    Profile(id, GetName(), m_parameters, m_return_type);
  }

  static void Profile(llvm::FoldingSetNodeID &id, ConstString func_name,
                      llvm::ArrayRef<CompilerType> parameters,
                      CompilerType return_type) {
    id.AddString(func_name.GetStringRef());
    id.AddInteger(parameters.size());

    if (return_type.IsValid())
      id.AddPointer(return_type.GetOpaqueQualType());

    for (const auto &param : parameters)
      id.AddPointer(param.GetOpaqueQualType());
  }

private:
  llvm::SmallVector<CompilerType, 4> m_parameters;
  CompilerType m_return_type;
  ConstString m_function_signature;
};

} // namespace fortran
} // namespace plugin
} // namespace lldb_private

#endif // LLDB_SOURCE_PLUGINS_TYPESYSTEM_FORTRAN_FORTRANTYPES_H
