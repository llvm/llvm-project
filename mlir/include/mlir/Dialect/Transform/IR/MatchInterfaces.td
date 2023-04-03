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

def SingleOpMatcher : NativeOpTrait<"SingleOpMatcherOpTrait"> {
  let cppNamespace = "::mlir::transform";

  string extraDeclaration = [{
    ::mlir::DiagnosedSilenceableFailure matchOperation(
          ::mlir::Operation *current,
          ::mlir::transform::TransformResults &results,
          ::mlir::transform::TransformState &state);
  }];
}
