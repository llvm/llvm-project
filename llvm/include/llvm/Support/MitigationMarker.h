//===---- MitigationMarker.h - Emit LLVM Code from ASTs for a Module ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This enables tagging functions with metadata to indicate mitigations are
// applied to them.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_MITIGATIONMARKER_H
#define LLVM_SUPPORT_MITIGATIONMARKER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include <unordered_map>

namespace llvm {

enum class MitigationKey {
  AUTO_VAR_INIT = 0,

  STACK_CLASH_PROTECTION = 1,

  STACK_PROTECTOR = 2,
  STACK_PROTECTOR_STRONG = 3,
  STACK_PROTECTOR_ALL = 4,

  CFI_ICALL = 5,
  CFI_VCALL = 6,
  CFI_NVCALL = 7,

  MITIGATION_KEY_MAX
};

const llvm::DenseMap<MitigationKey, StringRef> &GetMitigationMetadataMapping();

} // namespace llvm

#endif // PIKA_SUPPORT_MITIGATIONMARKER_H
