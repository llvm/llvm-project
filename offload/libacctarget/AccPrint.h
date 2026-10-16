//===- AccPrint.h - Printing of OpenACC runtime values ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Printing utilities for OpenACC runtime values.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBACCTARGET_ACCPRINT_H_
#define _LIBACCTARGET_ACCPRINT_H_

#include "ISO_Fortran_binding.h"
#include "openacc.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace llvm::acc::print {

#include "AccPrintEnums.inc"

/// Prints a Fortran `logical(c_bool)`
struct Logical {
  bool Value;
};
inline Logical logical(bool V) { return {V}; }
inline raw_ostream &operator<<(raw_ostream &OS, Logical L) {
  return OS << (L.Value ? "true" : "false");
}

/// Prints an async argument, naming the special values (acc_async_sync, ...).
struct AsyncArg {
  int64_t Value;
};
inline AsyncArg async(int64_t V) { return {V}; }
inline raw_ostream &operator<<(raw_ostream &OS, AsyncArg A) {
  if (const char *Name = asyncConstName(A.Value))
    return OS << Name << " (" << A.Value << ")";
  return OS << A.Value;
}

inline const char *cfiAttributeName(CFI_attribute_t A) {
  switch (A) {
  case CFI_attribute_pointer:
    return "pointer";
  case CFI_attribute_allocatable:
    return "allocatable";
  case CFI_attribute_other:
    return "other";
  default:
    return "<unknown>";
  }
}

} // namespace llvm::acc::print

/// Prints OpenACC enums by name, or as `acc_device_t(42)` if the value is not
/// an enumerator.
template <typename E>
inline std::enable_if_t<std::is_same_v<E, acc_device_t> ||
                            std::is_same_v<E, acc_device_property_t>,
                        llvm::raw_ostream &>
operator<<(llvm::raw_ostream &OS, E V) {
  if (const char *Name = llvm::acc::print::enumName(V))
    return OS << Name;
  return OS << llvm::acc::print::enumTypeName(V) << "(" << static_cast<int>(V)
            << ")";
}

/// Pretty prints a Fortran descriptor.
inline llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                     const CFI_cdesc_t *Desc) {
  if (!Desc)
    return OS << "null";
  OS << static_cast<const void *>(Desc) << " {base=" << Desc->base_addr
     << ", rank=" << static_cast<int>(Desc->rank)
     << ", elem_len=" << Desc->elem_len
     << ", attr=" << llvm::acc::print::cfiAttributeName(Desc->attribute);
  int Rank = std::min<int>(Desc->rank, CFI_MAX_RANK);
  if (Rank > 0) {
    OS << ", dims=[";
    for (int I = 0; I < Rank; ++I) {
      const CFI_dim_t &D = Desc->dim[I];
      OS << (I ? ", " : "") << D.lower_bound << ":" << D.extent << ":" << D.sm;
    }
    OS << "]";
  }
  return OS << "}";
}

#endif // _LIBACCTARGET_ACCPRINT_H_
