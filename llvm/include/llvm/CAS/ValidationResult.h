//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CAS_VALIDATIONRESULT_H
#define LLVM_CAS_VALIDATIONRESULT_H

namespace llvm::cas {

/// Represents the result of validating the contents, e.g. using
/// \c validateOnDiskUnifiedCASDatabasesIfNeeded, or of recovering from a
/// failed validation, e.g. using \c recoverOnDiskUnifiedCASDatabases.
///
/// Note: invalid results are handled as an \c Error.
enum class ValidationResult {
  /// The data is already valid.
  Valid,
  /// The data was invalid, but was recovered.
  Recovered,
  /// Validation or recovery was skipped, as it was not needed.
  Skipped,
};

} // namespace llvm::cas

#endif // LLVM_CAS_VALIDATIONRESULT_H
