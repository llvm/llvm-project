//===- Materialize.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_CVPACK_MATERIALIZE_H
#define LLVM_DEBUGINFO_CVPACK_MATERIALIZE_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace llvm {
namespace codeview {
namespace cvpack {

class CVPackResolver;

/// Restore the original COFF bytes of a packed object, and check them against
/// the hash that its origin record holds. The pack that its LF_CVPACK_LIST
/// names is searched in ObjectDirectories, usually the directory of the
/// object, then in the directories of the resolver.
LLVM_ABI Expected<std::vector<uint8_t>>
materializeObject(ArrayRef<uint8_t> Object, CVPackResolver &Resolver,
                  ArrayRef<StringRef> ObjectDirectories = {});

} // namespace cvpack
} // namespace codeview
} // namespace llvm

#endif // LLVM_DEBUGINFO_CVPACK_MATERIALIZE_H
