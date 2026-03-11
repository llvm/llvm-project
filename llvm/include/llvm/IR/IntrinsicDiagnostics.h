//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines IntrinsicDiagnosticsProvider, a virtual base class that targets
// can subclass to append extra detail lines to intrinsic signature mismatch
// diagnostics.  The default implementations are no-ops, leaving the standard
// upstream message unchanged.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_INTRINSICDIAGNOSTICS_H
#define LLVM_IR_INTRINSICDIAGNOSTICS_H

#include "llvm/ADT/StringRef.h"

namespace llvm {
class FunctionType;
class raw_ostream;

/// Interface for appending target-specific detail lines to diagnostics emitted
/// when an intrinsic is called or declared with an incorrect signature.
///
/// Subclasses are registered via \c registerProvider.  At diagnostic emission
/// time each registered provider may write additional lines to the output
/// stream; the standard upstream message is always emitted first by the caller.
class IntrinsicDiagnosticsProvider {
public:
  virtual ~IntrinsicDiagnosticsProvider() = default;

  /// Optionally appends detail for a call-site signature mismatch.
  virtual void getSignatureMismatch(StringRef IntrName, FunctionType *DeclFTy,
                                    FunctionType *CallFTy,
                                    raw_ostream &OS) const {}

  /// Optionally appends detail for a verifier return-type mismatch.
  virtual void getReturnTypeMismatch(StringRef IntrName, FunctionType *IFTy,
                                     raw_ostream &OS) const {}

  /// Optionally appends detail for a verifier argument-type mismatch.
  virtual void getArgTypeMismatch(StringRef IntrName, FunctionType *IFTy,
                                  raw_ostream &OS) const {}

  /// Optionally appends detail for a parser-level signature mismatch.
  /// \p ExpectedFTy is the canonical intrinsic type, or null for overloaded
  /// intrinsics.
  virtual void getParserMismatch(StringRef IntrName, FunctionType *CallFTy,
                                 FunctionType *ExpectedFTy,
                                 raw_ostream &OS) const {}

  /// Registers \p P as a diagnostics provider.  Does not take ownership;
  /// \p P must outlive all diagnostic queries.
  static void registerProvider(IntrinsicDiagnosticsProvider *P);

  /// Asks all registered providers to append detail to \p OS.
  static void querySignatureMismatch(StringRef IntrName, FunctionType *DeclFTy,
                                     FunctionType *CallFTy, raw_ostream &OS);
  static void queryReturnTypeMismatch(StringRef IntrName, FunctionType *IFTy,
                                      raw_ostream &OS);
  static void queryArgTypeMismatch(StringRef IntrName, FunctionType *IFTy,
                                   raw_ostream &OS);
  static void queryParserMismatch(StringRef IntrName, FunctionType *CallFTy,
                                  FunctionType *ExpectedFTy, raw_ostream &OS);
};

} // namespace llvm

#endif // LLVM_IR_INTRINSICDIAGNOSTICS_H
