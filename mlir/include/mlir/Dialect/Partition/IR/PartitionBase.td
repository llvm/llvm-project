 //===- PartitionBase.td - Partition dialect base ----*- tablegen -*-===//
 //
 // Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 // See https://llvm.org/LICENSE.txt for license information.
 // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 //
 //===----------------------------------------------------------------------===//

 #ifndef PARTITION_BASE
 #define PARTITION_BASE

 include "mlir/IR/OpBase.td"

 def Partition_Dialect : Dialect {
   let name = "partition";
   let cppNamespace = "::mlir::partition";
   let description = [{
     The `partition` dialect is intended to hold primitives that
     form a bridge between high-level operations on partition tensors
     and lower-level operations on the actual partition schemes
     consisting of primary tensor, indices, and data. This bridge
     simplifies a `partition compiler` pass by postponing actual
     code generation for the supported primitives to a later phase,
     either by generating calls into a runtime support library
     or by further lowering the primitives into actual code.
   }];
 }

 #endif // PARTITION_BASE
