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

class GENOpTrait<string name, list<Trait> traits = [],
                 code extraOpDeclaration = [{}],
                 code extraOpDefinition = [{}]>
    : NativeOpTrait<name, traits, extraOpDeclaration, extraOpDefinition> {
  let cppNamespace = "::mlir::OpTrait::GEN";
}

//===----------------------------------------------------------------------===//
// ND-Range Operations
//===----------------------------------------------------------------------===//

def GEN3DNDRange : GENOpTrait<"GEN3DNDRange">;

class GEN_3DNDRangeOp<string mnemonic, list<Trait> traits = []>
    : GEN_Op<mnemonic, [GEN3DNDRange, Pure] # traits>,
      Arguments<(ins I32:$dim)>,
      Results<(outs Index:$res)> {
  let assemblyFormat = "$dim attr-dict";
}

def GEN_LocalIdOp : GEN_3DNDRangeOp<"local_id"> {
  let summary = "Query a work-item's local id.";
  let description = [{
    Query the work-item's position in its work-group, i.e., its local id, in a
    given dimension.
    ```mlir
    %local_id = gen.local_id %dim
    ```
  }];
}

def GEN_WorkGroupIdOp : GEN_3DNDRangeOp<"work_group_id"> {
  let summary = "Query the id of a work-item's work-group.";
  let description = [{
    Query the id of a work-item's work-group in a given dimension.
    ```mlir
    %work_group_id = gen.work_group_id %dim
    ```
  }];
}

def GEN_WorkGroupSizeOp : GEN_3DNDRangeOp<"work_group_size"> {
  let summary = "Query the work-group size.";
  let description = [{
    Query the work-item's work-group size in a given dimension.
    ```mlir
    %work_group_size = gen.work_group_size %dim
    ```
  }];
}

def GEN_NumWorkGroupsOp : GEN_3DNDRangeOp<"num_work_groups"> {
  let summary = "Query the number of work-groups in the ND-range.";
  let description = [{
    Query the number of work-groups in the ND-range in a given dimension.
    ```mlir
    %wg_number = gen.num_work_groups %dim
    ```
  }];
}

//===----------------------------------------------------------------------===//
// Synchronization
//===----------------------------------------------------------------------===//

def GEN_BarrierOp : GEN_Op<"barrier"> {
  let summary = "Work-group barrier";

  let description = [{
    The `gen.barrier` operation performs a work-group barrier and ensures all
    outstanding memory transaction using local or global memory are complete.
  }];

  let assemblyFormat = "attr-dict";
}

def IntegerOrFloatType : AnyTypeOf<[AnySignlessInteger, AnyFloat]>;

def GEN_SubGroupShuffleOp
    : GEN_Op<"sub_group_shuffle", [Pure, AllTypesMatch<["res", "value"]>]>,
  Results<(outs IntegerOrFloatType:$res)>,
  Arguments<(ins IntegerOrFloatType:$value,
                 I32:$mask,
                 GEN_ShflKindAttr:$kind)> {
  let summary = "Sub-group shuffle";
  let description = [{
    The `gen.sub_group_shuffle` operation is invoked by different work items
    with different values, given by $value. Different work items have different
    sub-group local IDs. The shuffle kind, $kind, is given to determine how to
    calculate the associated sub-group local ID. It returns the associated
    $value for the work item with sub-group local ID equal to:
    - $kind == xor, the current invocation’s sub-group local ID xor’ed with $mask.
    - $kind == up, the current invocation’s sub-group local ID - $mask.
    - $kind == down, the current invocation’s sub-group local ID + $mask.
    - $kind == idx, the sub-group local ID $mask.
  }];

  let assemblyFormat = [{
    $kind $value `,` $mask attr-dict `:` type($res)
  }];
}

#endif // GEN_OPS
