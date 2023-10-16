// ===- PartitionDialect.cpp - partition dialect implementation -----===//
 //
 // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 // See https://llvm.org/LICENSE.txt for license information.
 // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 //
 //===----------------------------------------------------------------------===//

 #include "mlir/Dialect/Partition/IR/Partition.h"

 #include "mlir/IR/Builders.h"
 #include "mlir/IR/OpImplementation.h"

 using namespace mlir;
 using namespace mlir::partition;

 void PartitionDialect::initialize() {
   addOperations<
 #define GET_OP_LIST
 #include "mlir/Dialect/Partition/IR/PartitionOps.cpp.inc"
       >();
 }

 #define GET_OP_CLASSES
 #include "mlir/Dialect/Partition/IR/PartitionOps.cpp.inc"
