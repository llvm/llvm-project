//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DEBUGINFO_BTF_BTFTYPEUTILS_H
#define LLVM_LIB_DEBUGINFO_BTF_BTFTYPEUTILS_H

#include "llvm/DebugInfo/BTF/BTF.h"
#include "llvm/Support/Error.h"

namespace llvm::BTF {

// Validate all references in a complete, native-endian type record. Zero is
// reserved for void; every other reference must belong to this type table.
Error validateTypeRefs(const CommonType &Type, uint32_t MaxId);

} // namespace llvm::BTF

#endif
