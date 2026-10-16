//===- DeviceMappingInterface.td - Device mapping interfaces*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the interfaces for the device mapping specification for the loops.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DEVICEMAPPINGINTERFACE
#define MLIR_DEVICEMAPPINGINTERFACE

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Attribute interfaces
//===----------------------------------------------------------------------===//

def DeviceMappingAttrInterface : AttrInterface<"DeviceMappingAttrInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Attribute interface describing how to map a region to a processing unit.

    It is intended to be a generic mechanism for binding regions to execution
    units of an actual or virtual device. Each device first expresses its own
    mappings, and those mappings must implement this interface. These mappings
    can be used by the device-specific code generators and the desired regions
    can be connected to the given processing unit.

    Currently, `scf.forall` uses this interface to express the mapping
    of the loops it contains to the GPU's parallelism units such as threads and
    thread blocks.
  }];

 let methods = [
    InterfaceMethod<
      /*desc=*/"Return mapping as an integer from the attribute.",
      /*retTy=*/"int64_t",
      /*methodName=*/"getMappingId",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Return true if the attribute specifies a linear mapping",
      /*retTy=*/"bool",
      /*methodName=*/"isLinearMapping",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the [0..n) relative index of the attribute depending on its group.
        This can be used to index into a contiguous array.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getRelativeIndex", 
      /*args=*/(ins)
    >
  ];
}

def DeviceMaskingAttrInterface : AttrInterface<"DeviceMaskingAttrInterface"> {
  let cppNamespace = "::mlir";
  let description = [{
    Attribute interface describing how to filter the processing units that a
    region is mapped to.

    For instance, consider the following example mask which specifies processing
    units 2, 4 and 5 are active:
    ```
              8       4       0
      mask  : 0 0 0 1 1 0 1 0 0
    ```
    The logical ID for an active processing unit is defined as its position
    relative to the other active processing units. In this example, we have:
    ```
      Processing Unit    LogicalID
              0             N/A
              1             N/A
              2              0
              3             N/A
              4              1
              5              2
              6             N/A
              7             N/A
    ```
  }];

 let methods = [
    InterfaceMethod<
      /*desc=*/[{
        Create the logical active id for a given physical id.
        Expects a physicalLinearMappingId of I64Type.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"createLogicalLinearMappingId",
      /*args=*/(ins "OpBuilder&":$builder, "Value":$physicalLinearMappingId)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the dynamic condition determining whether a given physical id is
        active under the mask.
        Expects a physicalLinearMappingId of I64Type.
      }],
      /*retTy=*/"Value",
      /*methodName=*/"createIsActiveIdPredicate",
      /*args=*/(ins "OpBuilder&":$builder, "Value":$physicalLinearMappingId)
    >,
    InterfaceMethod<
      /*desc=*/[{
        Return the maximal number of pysical ids supported.
        This is to account for temporary implementation limitations (e.g. i64)
        and fail gracefully with actionnable error messages.
      }],
      /*retTy=*/"int64_t",
      /*methodName=*/"getMaxNumPhysicalIds",
      /*args=*/(ins)
    >,
  ];
}

def DeviceMappingArrayAttr :
  TypedArrayAttrBase<AnyAttrOf<[DeviceMappingAttrInterface, DeviceMaskingAttrInterface]>,
  "Device Mapping array attribute"> { }

#endif // MLIR_DEVICEMAPPINGINTERFACE
