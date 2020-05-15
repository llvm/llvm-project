//===- BPFCORE.h - Common info for Compile-Once Run-EveryWhere  -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_BPFCORE_H
#define LLVM_LIB_TARGET_BPF_BPFCORE_H

namespace llvm {

class BPFCoreSharedInfo {
public:
  enum PatchableRelocKind : uint32_t {
    FIELD_BYTE_OFFSET = 0,
    FIELD_BYTE_SIZE,
    FIELD_EXISTENCE,
    FIELD_SIGNEDNESS,
    FIELD_LSHIFT_U64,
    FIELD_RSHIFT_U64,
    BTF_TYPE_ID_LOCAL,
    BTF_TYPE_ID_REMOTE,

    MAX_FIELD_RELOC_KIND,
  };

  enum BTFTypeIdFlag : uint32_t {
    BTF_TYPE_ID_LOCAL_RELOC = 0,
    BTF_TYPE_ID_REMOTE_RELOC,

    MAX_BTF_TYPE_ID_FLAG,
  };

  /// The attribute attached to globals representing a field access
  static const std::string AmaAttr;
  /// The attribute attached to globals representing a type id
  static const std::string TypeIdAttr;
};

} // namespace llvm

#endif
