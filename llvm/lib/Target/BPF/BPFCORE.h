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
  /// The attribute attached to globals representing a member offset
  static const std::string AmaAttr;
  /// The section name to identify a patchable external global
  static const std::string PatchableExtSecName;
};

} // namespace llvm

#endif
