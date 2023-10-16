//===- Transforms.h - Partition transformations -------------*- C++ -*-===//
 //
 // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 // See https://llvm.org/LICENSE.txt for license information.
 // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 //
 //===----------------------------------------------------------------------===//

 #ifndef MLIR_DIALECT_PARTITION_TRANSFORMS_TRANSFORMS_H_
 #define MLIR_DIALECT_PARTITION_TRANSFORMS_TRANSFORMS_H_

 #include "mlir/IR/PatternMatch.h"

 namespace mlir {
 namespace partition {

 /// Sets up sparsification conversion rules with the given options.
 void populateSparsificationConversionPatterns(RewritePatternSet &patterns);

 } // namespace partition
 } // namespace mlir

 #endif // MLIR_DIALECT_PARTITION_TRANSFORMS_TRANSFORMS_H_
