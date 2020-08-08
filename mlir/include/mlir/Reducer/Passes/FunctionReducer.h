//===- FunctionReducer.h - MLIR Reduce Function Reducer ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the FunctionReducer class. It defines a variant generator
// method with the purpose of producing different variants by eliminating
// functions from the  parent module.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_REDUCER_PASSES_FUNCTIONREDUCER_H
#define MLIR_REDUCER_PASSES_FUNCTIONREDUCER_H

#include "mlir/Reducer/ReductionNode.h"
#include "mlir/Reducer/Tester.h"

namespace mlir {

/// The FunctionReducer class defines a variant generator method that produces
/// multiple variants by eliminating different operations from the
/// parent module.
class FunctionReducer {
public:
  /// Generate variants by removing functions from the module in the parent
  /// Reduction Node and link the variants as children in the Reduction Tree
  /// Pass.
  void generateVariants(ReductionNode *parent, const Tester *test);
};

} // end namespace mlir

#endif
