//===- ABIRewriteContext.h - Dialect-specific ABI rewriting -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines ABIRewriteContext, the abstract interface for dialect-
// specific ABI lowering rewrites.  Each MLIR dialect that wants ABI lowering
// (CIR, FIR, etc.) provides a concrete subclass.
//
// ABIRewriteContext consumes ABI classification results and drives the
// creation of lowered function signatures, argument coercions, and call
// site rewrites using dialect-specific operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ABI_ABIREWRITECONTEXT_H
#define MLIR_ABI_ABIREWRITECONTEXT_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/Support/Alignment.h"

namespace mlir {
namespace abi {

/// Classification of how a single argument or return value should be
/// passed at the ABI level.
///
/// This is a dialect-agnostic representation.  It mirrors the kinds
/// found in the LLVM ABI library and in CIR's ABIArgInfo, but does
/// not depend on either.
enum class ArgKind : uint8_t {
  /// Pass directly in registers, possibly coerced to a different type.
  Direct,

  /// Like Direct, but with a sign/zero extension attribute.
  Extend,

  /// Pass indirectly via a pointer (sret for returns, byval for args).
  Indirect,

  /// Ignore (void return, empty struct).
  Ignore,

  /// Expand an aggregate into its constituent scalar fields.
  Expand,
};

/// Describes how a single argument or return value is passed after ABI
/// lowering.
struct ArgClassification {
  ArgKind kind = ArgKind::Direct;

  /// The ABI-coerced type, if different from the original.  Null means
  /// use the original type.
  Type coercedType = nullptr;

  /// For Indirect: alignment of the pointed-to object.
  llvm::Align indirectAlign = llvm::Align(1);

  /// For Extend: whether to sign-extend (true) or zero-extend (false).
  bool signExtend = false;

  /// For Direct: whether a struct coercion can be flattened into
  /// individual register-width arguments.
  bool canFlatten = true;

  /// For Indirect: whether the callee gets ownership (byval).
  bool byVal = false;

  static ArgClassification getDirect(Type coerced = nullptr) {
    ArgClassification c;
    c.kind = ArgKind::Direct;
    c.coercedType = coerced;
    return c;
  }

  static ArgClassification getIgnore() {
    ArgClassification c;
    c.kind = ArgKind::Ignore;
    return c;
  }

  static ArgClassification getIndirect(llvm::Align align, bool byVal = true) {
    ArgClassification c;
    c.kind = ArgKind::Indirect;
    c.indirectAlign = align;
    c.byVal = byVal;
    return c;
  }

  static ArgClassification getExtend(Type coerced, bool signExt) {
    ArgClassification c;
    c.kind = ArgKind::Extend;
    c.coercedType = coerced;
    c.signExtend = signExt;
    return c;
  }
};

/// Holds the full ABI classification for a function: return type and
/// all arguments.
struct FunctionClassification {
  ArgClassification returnInfo;
  SmallVector<ArgClassification> argInfos;
};

/// ABIRewriteContext is the abstract interface that each dialect
/// implements to perform ABI-specific rewrites on its operations.
///
/// The pass orchestrator calls these methods after ABI classification
/// to rewrite function definitions and call sites.
class ABIRewriteContext {
public:
  virtual ~ABIRewriteContext() = default;

  /// Rewrite a function definition to use ABI-lowered types.
  ///
  /// This creates a new function with the lowered signature, rewrites
  /// the function body to adapt between the ABI types and the
  /// original high-level types, and replaces the original function.
  ///
  /// \param funcOp  The function to rewrite (via FunctionOpInterface).
  /// \param fc      The ABI classification for this function.
  /// \param rewriter  The pattern rewriter to use for modifications.
  /// \returns success() if the function was rewritten.
  virtual LogicalResult
  rewriteFunctionDefinition(FunctionOpInterface funcOp,
                            const FunctionClassification &fc,
                            OpBuilder &rewriter) = 0;

  /// Rewrite a call operation to match the callee's ABI-lowered
  /// signature.
  ///
  /// This coerces arguments, handles indirect returns (sret), and
  /// adapts the call result back to the original high-level type.
  ///
  /// \param callOp  The call operation to rewrite.
  /// \param fc      The ABI classification for the callee.
  /// \param rewriter  The pattern rewriter to use for modifications.
  /// \returns success() if the call was rewritten.
  virtual LogicalResult rewriteCallSite(Operation *callOp,
                                        const FunctionClassification &fc,
                                        OpBuilder &rewriter) = 0;

  /// Return the dialect namespace this context handles (e.g. "cir").
  virtual StringRef getDialectNamespace() const = 0;
};

} // namespace abi
} // namespace mlir

#endif // MLIR_ABI_ABIREWRITECONTEXT_H
