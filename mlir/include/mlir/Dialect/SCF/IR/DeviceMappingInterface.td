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

    Currently, `scf.foreach_thread` uses this interface to express the mapping
    of the loops it contains to the GPU's parallelism units such as threads and
    thread blocks.
  }];

 let methods = [
    InterfaceMethod<[{
        Returns mapping as an integer from the attribute.
      }],
      "int64_t", "getMappingId", (ins)
    >
  ];
}

def DeviceMappingArrayAttr :
  TypedArrayAttrBase<DeviceMappingAttrInterface,
  "Device Mapping array attribute"> { }

#endif // MLIR_DEVICEMAPPINGINTERFACE
