//===-- Passes.td - ArmSVE pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def LegalizeVectorStorage
    : Pass<"arm-sve-legalize-vector-storage", "mlir::func::FuncOp"> {
  let summary = "Ensures stores of SVE vector types will be legal";
  let description = [{
    This pass ensures that loads, stores, and allocations of SVE vector types
    will be legal in the LLVM backend. It does this at the memref level, so this
    pass must be applied before lowering all the way to LLVM.

    This pass currently addresses two issues.

    #### Loading and storing predicate types

    It is only legal to load/store predicate types equal to (or greater than) a
    full predicate register, which in MLIR is `vector<[16]xi1>`. Smaller
    predicate types (`vector<[1|2|4|8]xi1>`) must be converted to/from a full
    predicate type (referred to as a `svbool`) before and after storing and
    loading respectively. This pass does this by widening allocations and
    inserting conversion intrinsics. Note: Non-powers-of-two masks (e.g.
    `vector<[7]xi1>`), which are not SVE predicates, are ignored.

    For example:

    ```mlir
    %alloca = memref.alloca() : memref<vector<[4]xi1>>
    %mask = vector.constant_mask [4] : vector<[4]xi1>
    memref.store %mask, %alloca[] : memref<vector<[4]xi1>>
    %reload = memref.load %alloca[] : memref<vector<[4]xi1>>
    ```
    Becomes:
    ```mlir
    %alloca = memref.alloca() {alignment = 1 : i64} : memref<vector<[16]xi1>>
    %mask = vector.constant_mask [4] : vector<[4]xi1>
    %svbool = arm_sve.convert_to_svbool %mask : vector<[4]xi1>
    memref.store %svbool, %alloca[] : memref<vector<[16]xi1>>
    %reload_svbool = memref.load %alloca[] : memref<vector<[16]xi1>>
    %reload = arm_sve.convert_from_svbool %reload_svbool : vector<[4]xi1>
    ```

    #### Relax alignments for SVE vector allocas

    The storage for SVE vector types only needs to have an alignment that
    matches the element type (for example 4 byte alignment for `f32`s). However,
    the LLVM backend currently defaults to aligning to `base size` x
    `element size` bytes. For non-legal vector types like `vector<[8]xf32>` this
    results in 8 x 4 = 32-byte alignment, but the backend only supports up to
    16-byte alignment for SVE vectors on the stack. Explicitly setting a smaller
    alignment prevents this issue.
  }];
  let constructor = "mlir::arm_sve::createLegalizeVectorStoragePass()";
  let dependentDialects = ["func::FuncDialect",
    "memref::MemRefDialect", "vector::VectorDialect",
    "arm_sve::ArmSVEDialect"];
}

#endif // MLIR_DIALECT_ARMSVE_TRANSFORMS_PASSES_TD
