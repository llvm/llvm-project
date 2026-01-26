//===- OffloadYAML.h - Offload Binary YAMLIO implementation -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares classes for handling the YAML representation of
/// offloading binaries.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECTYAML_OFFLOADYAML_H
#define LLVM_OBJECTYAML_OFFLOADYAML_H

#include "llvm/Object/OffloadBinary.h"
#include "llvm/ObjectYAML/YAML.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/YAMLTraits.h"
#include <optional>

namespace llvm {
namespace OffloadYAML {

struct Binary {
  struct StringEntry {
    StringRef Key;
    StringRef Value;
  };

  struct Member {
    std::optional<object::ImageKind> ImageKind;
    std::optional<object::OffloadKind> OffloadKind;
    std::optional<uint32_t> Flags;
    std::optional<std::vector<StringEntry>> StringEntries;
    std::optional<yaml::BinaryRef> Content;
  };

  std::optional<uint32_t> Version;
  std::optional<uint64_t> Size;
  std::optional<uint64_t> EntryOffset;
  std::optional<uint64_t> EntrySize;
  std::vector<Member> Members;
};

} // end namespace OffloadYAML
} // end namespace llvm

LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::OffloadYAML::Binary::Member)
LLVM_YAML_IS_SEQUENCE_VECTOR(llvm::OffloadYAML::Binary::StringEntry)

namespace llvm {
namespace yaml {

template <> struct ScalarEnumerationTraits<object::ImageKind> {
  LLVM_ABI static void enumeration(IO &IO, object::ImageKind &Value);
};

template <> struct ScalarEnumerationTraits<object::OffloadKind> {
  LLVM_ABI static void enumeration(IO &IO, object::OffloadKind &Value);
};

template <> struct MappingTraits<OffloadYAML::Binary> {
  LLVM_ABI static void mapping(IO &IO, OffloadYAML::Binary &O);
};

template <> struct MappingTraits<OffloadYAML::Binary::StringEntry> {
  LLVM_ABI static void mapping(IO &IO, OffloadYAML::Binary::StringEntry &M);
};

template <> struct MappingTraits<OffloadYAML::Binary::Member> {
  LLVM_ABI static void mapping(IO &IO, OffloadYAML::Binary::Member &M);
};

} // end namespace yaml
} // end namespace llvm

#endif // LLVM_OBJECTYAML_OFFLOADYAML_H
