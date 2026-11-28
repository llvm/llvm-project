//===- CodeViewYAMLPacks.h - CodeView YAML for CodeView packs ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// YAML traits for the payloads of the CodeView pack records
// (LF_CVPACK_LIST, LF_CVPACK_ORIGIN, LF_CVPACK_MAP and S_CVPACK_MAP).
// The records themselves are encoded by llvm/DebugInfo/CodeView/CVPacks.h.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_CODEVIEWYAMLCVPACKS_H
#define LLVM_OBJECTYAML_CODEVIEWYAMLCVPACKS_H

#include "llvm/DebugInfo/CodeView/CVPacks.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/YAMLTraits.h"

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::codeview::cvpack::RangeMapEntry)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::codeview::cvpack::OriginEdit)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::codeview::cvpack::RestorationFragment)

namespace llvm {
namespace yaml {

/// A content id, as 64 lowercase hex digits.
template <> struct ScalarTraits<codeview::cvpack::PackId> {
  LLVM_ABI static void output(const codeview::cvpack::PackId &Value, void *,
                              raw_ostream &OS);
  LLVM_ABI static StringRef input(StringRef Scalar, void *,
                                  codeview::cvpack::PackId &Value);
  static QuotingType mustQuote(StringRef) { return QuotingType::None; }
};

template <> struct ScalarEnumerationTraits<codeview::cvpack::OriginEditKind> {
  LLVM_ABI static void enumeration(IO &IO,
                                   codeview::cvpack::OriginEditKind &Value);
};

template <> struct MappingTraits<codeview::cvpack::RangeMapEntry> {
  LLVM_ABI static void mapping(IO &IO, codeview::cvpack::RangeMapEntry &Entry);
};

template <> struct MappingTraits<codeview::cvpack::RestorationFragment> {
  LLVM_ABI static void mapping(IO &IO,
                               codeview::cvpack::RestorationFragment &Fragment);
};

template <> struct MappingTraits<codeview::cvpack::OriginEdit> {
  LLVM_ABI static void mapping(IO &IO, codeview::cvpack::OriginEdit &Edit);
};

template <> struct MappingTraits<codeview::cvpack::Origin> {
  LLVM_ABI static void mapping(IO &IO, codeview::cvpack::Origin &Data);
};

} // namespace yaml
} // namespace llvm

#endif // LLVM_OBJECTYAML_CODEVIEWYAMLCVPACKS_H
