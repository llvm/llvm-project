//===-- GENOps.td - GEN IR dialect op definition file ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the GEN IR operation definition file.
//
//===----------------------------------------------------------------------===//

#ifndef GEN_OPS
#define GEN_OPS

include "mlir/Dialect/GEN/IR/GENDialect.td"
include "mlir/Dialect/GEN/IR/GENAttrDefs.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

//===----------------------------------------------------------------------===//
// GEN op definitions
//===----------------------------------------------------------------------===//

class GEN_Op<string mnemonic, list<Trait> traits = []> :
    Op<GEN_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ND-Range Operations
//===----------------------------------------------------------------------===//

class GEN_3DNDRangeOp<string mnemonic, list<Trait> traits = []>
    : GEN_Op<mnemonic, [NoMemoryEffect] # traits> {
  let arguments = (ins I32:$dim);
  let results = (outs Index:$res);

  let assemblyFormat = "$dim attr-dict";
}

def GEN_LocalIdOp : GEN_3DNDRangeOp<"local_id"> {
  let summary = "Query a work-item's local id.";
  let description = [{
    Query the work-item's position in its work-group, i.e., its local id, in a
    given dimension `dim`, which must be either 0, 1 or 2. Behavior is undefined
    for invalid `dim` values.

    Example:
    ```mlir
    %local_id = gen.local_id %dim
    ```
  }];
}

def GEN_WorkGroupIdOp : GEN_3DNDRangeOp<"work_group_id"> {
  let summary = "Query the id of a work-item's work-group.";
  let description = [{
    Query the id of a work-item's work-group in a given dimension `dim`, which
    must be either 0, 1 or 2. Behavior is undefined for invalid `dim` values.

    Example:
    ```mlir
    %work_group_id = gen.work_group_id %dim
    ```
  }];
}

def GEN_WorkGroupSizeOp : GEN_3DNDRangeOp<"work_group_size"> {
  let summary = "Query the work-group size.";
  let description = [{
    Query the work-item's work-group size in a given dimension `dim`, which must
    be either 0, 1 or 2. Behavior is undefined for invalid `dim` values.

    Example:
    ```mlir
    %work_group_size = gen.work_group_size %dim
    ```
  }];
}

def GEN_NumWorkGroupsOp : GEN_3DNDRangeOp<"num_work_groups"> {
  let summary = "Query the number of work-groups in the ND-range.";
  let description = [{
    Query the number of work-groups in the ND-range in a given dimension `dim`,
    which must be either 0, 1 or 2. Behavior is undefined for invalid `dim`
    values.

    Example:
    ```mlir
    %wg_number = gen.num_work_groups %dim
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def GEN_BarrierOp : GEN_Op<"barrier"> {
  let summary = "Synchronizes all work-items of a work-group.";

  let description = [{
    Wait for all work-items in a given work-group to reach this execution point.
    All memory accesses made by these work-items prior to the operation are
    visible to all work-items in the work-group.

    It is undefined behavior unless none or all work-items in the work-group
    reach this execution point.

    Example:
    ```mlir
    gen.barrier
    ```
  }];

  let assemblyFormat = "attr-dict";
}

def ShuffleValueType
    : AnyTypeOf<[SignlessIntOfWidths<[8, 16, 32, 64]>, FloatOfWidths<[16, 32, 64]>]>;

def GEN_SubGroupShuffleOp
    : GEN_Op<"sub_group_shuffle", [Pure, AllTypesMatch<["res", "value"]>]> {
  let summary = "Sub-group shuffle";
  let description = [{
    The `gen.sub_group_shuffle` operation is invoked by different work items
    with different values, given by `value`. Different work items have different
    sub-group local IDs. The shuffle kind, `kind`, is given to determine how to
    calculate the associated sub-group local ID. It returns the associated
    `value` for the work item with sub-group local ID equal to:
    - `kind` == xor, the current invocation’s sub-group local ID xor'ed with `mask`.
    - `kind` == up, the current invocation’s sub-group local ID - `mask`.
    - `kind` == down, the current invocation’s sub-group local ID + `mask`.
    - `kind` == idx, the sub-group local ID `mask`.

    `value` and `res` types must match and can be any of: `i8`, `i16`, `i32`,
    `i64`, `f16`, `f32` or `f64`.

    Example:
    ```mlir
    // xor shuffle
    %0 = gen.sub_group_shuffle xor %arg0, %arg4 : i32

    // up shuffle
    %1 = gen.sub_group_shuffle up %arg1, %arg4 : i64

    // down shuffle
    %2 = gen.sub_group_shuffle down %arg2, %arg4 : f32

    // idx shuffle
    %3 = gen.sub_group_shuffle idx %arg3, %arg4 : f64
    ```
  }];

  let arguments = (ins ShuffleValueType:$value,
                       I32:$mask,
                       GEN_ShflKindAttr:$kind);
  let results = (outs ShuffleValueType:$res);

  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($res)
  }];
}

#endif // GEN_OPS
