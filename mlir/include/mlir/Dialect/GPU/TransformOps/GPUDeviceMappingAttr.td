//===-- GPUDeviceMappingAttr.td - Attribute definition -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the attribute used to map loops to gpu.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_DEVICE_MAPPING_ATTR
#define GPU_DEVICE_MAPPING_ATTR

include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"

def DimX : I64EnumAttrCase<"DimX", 0, "x">;
def DimY : I64EnumAttrCase<"DimY", 1, "y">;
def DimZ : I64EnumAttrCase<"DimZ", 2, "z">;

def ThreadsEnum : I64EnumAttr<"Threads", "threads for loop mapping", [
    DimX, DimY, DimZ]> {
  let cppNamespace = "::mlir::gpu";
}

def GPUThreadMappingAttr
    : GPU_Attr<"GPUThreadMapping", "thread", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ]> {
  let parameters = (ins
    EnumParameter<ThreadsEnum>:$thread
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining thread parallelism for GPU devices.

    Thread (aka work item) are grouped into a thread blocks where block may be
    described by a 1-, 2-, or 3-dimensional rectangle. This attribute indicates
    that thread parallelism is desired. It can be consumed by lowering to
    generate GPU.
  }];
}

def WarpsEnum : I64EnumAttr<"Warps", "threads for loop mapping", [
    DimX, DimY, DimZ]> {
  let cppNamespace = "::mlir::gpu";
}

def GPUWarpMappingAttr : GPU_Attr<"GPUWarpMapping", "warp", [
  DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ] >  {
  let parameters = (ins
    EnumParameter<WarpsEnum>:$warp
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining thread block parallelism for GPU devices.

    Warp (aka subgroup) are grouped into a grid where grid may be
    described by a 1-, 2-, or 3-dimensional rectangle. This attribute indicates
    that thread block parallelism is desired. It can be consumed by lowering to
    generate GPU code.
  }];
}

def BlocksEnum : I64EnumAttr<"Blocks", "threads for loop mapping", [
    DimX, DimY, DimZ]> {
  let cppNamespace = "::mlir::gpu";
}

def GPUBlockMappingAttr : GPU_Attr<"GPUBlockMapping", "block", [
  DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ] >  {
  let parameters = (ins
    EnumParameter<BlocksEnum>:$block
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining thread block parallelism for GPU devices.

    Thread blocks (aka work-group) are grouped into a grid where grid may be
    described by a 1-, 2-, or 3-dimensional rectangle. This attribute indicates
    that thread block parallelism is desired. It can be consumed by lowering to
    generate GPU code.
  }];
}

#endif // GPU_DEVICE_MAPPING_ATTR
