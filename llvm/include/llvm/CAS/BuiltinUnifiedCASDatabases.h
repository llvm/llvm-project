//===----------------------------------------------------------------------===//
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
LLVM_ABI
Expected<std::pair<std::unique_ptr<ObjectStore>, std::unique_ptr<ActionCache>>>
createOnDiskUnifiedCASDatabases(StringRef Path);

/// Represents the result of validating the contents using
/// \c validateOnDiskUnifiedCASDatabasesIfNeeded.
///
/// Note: invalid results are handled as an \c Error.
enum class ValidationResult {
  /// The data is already valid.
  Valid,
  /// The data was invalid, but was recovered.
  Recovered,
  /// Validation was skipped, as it was not needed.
  Skipped,
};

/// Validate the data in \p Path, if needed to ensure correctness.
///
/// \param Path directory for the on-disk database.
/// \param CheckHash Whether to validate hashes match the data.
/// \param AllowRecovery Whether to automatically recover from invalid data by
/// marking the files for garbage collection.
/// \param ForceValidation Whether to force validation to occur even if it
/// should not be necessary.
/// \param LLVMCasBinaryPath If provided, validation is performed out-of-process
/// using the given \c llvm-cas executable which protects against crashes
/// during validation. Otherwise validation is performed in-process.
///
/// \returns \c Valid if the data is already valid, \c Recovered if data
/// was invalid but has been cleared, \c Skipped if validation is not needed,
/// or an \c Error if validation cannot be performed or if the data is left
/// in an invalid state because \p AllowRecovery is false.
Expected<ValidationResult> validateOnDiskUnifiedCASDatabasesIfNeeded(
    StringRef Path, bool CheckHash, bool AllowRecovery, bool ForceValidation,
    std::optional<StringRef> LLVMCasBinaryPath);

} // namespace llvm::cas

#endif // LLVM_CAS_BUILTINUNIFIEDCASDATABASES_H
