//===- MatchInterfaces.td - Transform dialect interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"

def MatchOpInterface
    : OpInterface<"MatchOpInterface", [TransformOpInterface]> {
  let cppNamespace = "::mlir::transform";
}

// Trait for "matcher" transform operations that apply to an operation handle
// associated with at most one payload operation. Checks that it is indeed
// the case and produces a definite failure when it is not. The matching logic
// is implemented in the `matchOperation` function instead of `apply`. The op
// with this trait must provide a `Value getOperandHandle()` function that
// returns the handle to be used for matching.
def AtMostOneOpMatcher : NativeOpTrait<"AtMostOneOpMatcherOpTrait"> {
  let cppNamespace = "::mlir::transform";

  string extraDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure matchOperation(
          ::std::optional<::mlir::Operation *> maybeCurrent,
          ::mlir::transform::TransformResults &results,
          ::mlir::transform::TransformState &state);
  }];
}

// Trait for "matcher" transform operations that apply to an operation handle
// associated with exactly one payload operation. Checks that it is indeed
// the case and produces a definite failure when it is not. The matching logic
// is implemented in the `matchOperation` function instead of `apply`. The op
// with this trait must provide a `Value getOperandHandle()` function that
// returns the handle to be used for matching.
def SingleOpMatcher : NativeOpTrait<"SingleOpMatcherOpTrait"> {
  let cppNamespace = "::mlir::transform";

  string extraDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure matchOperation(
          ::mlir::Operation *current,
          ::mlir::transform::TransformResults &results,
          ::mlir::transform::TransformState &state);
  }];
}

// Trait for "matcher" transform operations that apply to a value handle
// associated with exactly one payload value. Checks that it is indeed
// the case and produces a definite failure when it is not. The matching logic
// is implemented in the `matchValue` function instead of `apply`. The op
// with this trait must provide a `Value getOperandHandle()` function that
// returns the handle to be used for matching.
def SingleValueMatcher : NativeOpTrait<"SingleValueMatcherOpTrait"> {
  let cppNamespace = "::mlir::transform";

  string extraDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure matchValue(
          ::mlir::Value current,
          ::mlir::transform::TransformResults &results,
          ::mlir::transform::TransformState &state);
  }];
}
