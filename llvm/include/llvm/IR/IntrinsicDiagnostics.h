//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines IntrinsicDiagnosticsProvider, a utility class with static methods
// that append detail lines to intrinsic signature mismatch diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICDIAGNOSTICS_H
#define LLVM_IR_INTRINSICDIAGNOSTICS_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class FunctionType;
class raw_ostream;

/// Utility class that formats detail for intrinsic signature mismatch
/// diagnostics.
class IntrinsicDiagnosticsProvider {
public:
  /// Appends detail for a call-site signature mismatch.
  LLVM_ABI static void querySignatureMismatch(StringRef IntrName,
                                              FunctionType *DeclFTy,
                                              FunctionType *CallFTy,
                                              raw_ostream &OS);

  /// Appends detail for a parser-level signature mismatch.
  /// \p ExpectedFTy is the canonical intrinsic type, or null for overloaded
  /// intrinsics.
  LLVM_ABI static void queryParserMismatch(StringRef IntrName,
                                           FunctionType *CallFTy,
                                           FunctionType *ExpectedFTy,
                                           raw_ostream &OS);
};

} // namespace llvm

#endif // LLVM_IR_INTRINSICDIAGNOSTICS_H
