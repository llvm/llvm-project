//===- BuiltinUnifiedCASDatabases.h -----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_BUILTINUNIFIEDCASDATABASES_H
#define LLVM_CAS_BUILTINUNIFIEDCASDATABASES_H

#include "llvm/Support/Error.h"

namespace llvm::cas {

class ActionCache;
class ObjectStore;

/// Create on-disk \c ObjectStore and \c ActionCache instances based on
/// \c ondisk::UnifiedOnDiskCache, with built-in hashing.
Expected<std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>>>
createOnDiskUnifiedCASDatabases(StringRef Path);

} // namespace llvm::cas

#endif // LLVM_CAS_BUILTINUNIFIEDCASDATABASES_H
