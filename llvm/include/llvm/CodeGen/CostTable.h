//===-- CostTable.h - Instruction Cost Table handling -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Cost tables and simple lookup functions
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_COSTTABLE_H_
#define LLVM_CODEGEN_COSTTABLE_H_

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGenTypes/MachineValueType.h"
#include <cstdint>

namespace llvm {

/// Cost Table Entry
template <typename CostType> struct CostTblEntryT {
  // Cost tables use aggregate initialization, so values that do not fit in
  // these fields are rejected as narrowing conversions at compile time.
  uint16_t ISD;
  uint8_t Type;
  CostType Cost;
};
using CostTblEntry = CostTblEntryT<unsigned>;
static_assert(sizeof(CostTblEntry) == 8);

/// Find in cost table.
template <class CostType>
inline const CostTblEntryT<CostType> *
CostTableLookup(ArrayRef<CostTblEntryT<CostType>> Tbl, int ISD, MVT Ty) {
  auto I = find_if(Tbl, [=](const CostTblEntryT<CostType> &Entry) {
    return ISD == Entry.ISD && Ty.SimpleTy == Entry.Type;
  });
  if (I != Tbl.end())
    return I;

  // Could not find an entry.
  return nullptr;
}

template <size_t N, class CostType>
inline const CostTblEntryT<CostType> *
CostTableLookup(const CostTblEntryT<CostType> (&Table)[N], int ISD, MVT Ty) {
  // Wrapper to fix template argument deduction failures.
  return CostTableLookup<CostType>(Table, ISD, Ty);
}

/// Type Conversion Cost Table
template <typename CostType> struct TypeConversionCostTblEntryT {
  uint16_t ISD;
  uint8_t Dst;
  uint8_t Src;
  CostType Cost;
};
using TypeConversionCostTblEntry = TypeConversionCostTblEntryT<unsigned>;
static_assert(sizeof(TypeConversionCostTblEntry) == 8);

/// Find in type conversion cost table.
template <class CostType>
inline const TypeConversionCostTblEntryT<CostType> *
ConvertCostTableLookup(ArrayRef<TypeConversionCostTblEntryT<CostType>> Tbl,
                       int ISD, MVT Dst, MVT Src) {
  auto I =
      find_if(Tbl, [=](const TypeConversionCostTblEntryT<CostType> &Entry) {
        return ISD == Entry.ISD && Src.SimpleTy == Entry.Src &&
               Dst.SimpleTy == Entry.Dst;
      });
  if (I != Tbl.end())
    return I;

  // Could not find an entry.
  return nullptr;
}

template <size_t N, class CostType>
inline const TypeConversionCostTblEntryT<CostType> *
ConvertCostTableLookup(const TypeConversionCostTblEntryT<CostType> (&Table)[N],
                       int ISD, MVT Dst, MVT Src) {
  // Wrapper to fix template argument deduction failures.
  return ConvertCostTableLookup<CostType>(Table, ISD, Dst, Src);
}

} // namespace llvm

#endif /* LLVM_CODEGEN_COSTTABLE_H_ */
