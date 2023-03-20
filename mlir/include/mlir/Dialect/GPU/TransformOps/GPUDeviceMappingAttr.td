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

def LinearIdEnum : I64EnumAttr<"LinearId", "linear ids for loop mapping", [
    DimX, DimY, DimZ]> {
  let cppNamespace = "::mlir::gpu";
}

def GPULinearIdMapping : GPU_Attr<"GPULinearIdMapping", "linear", [
  DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ] >  {
  let parameters = (ins
    EnumParameter<LinearIdEnum>:$linear_id
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute to allow re-interpreting the linear mapping for threads in GPU
    devices.

    Threads (aka work item) are grouped into a thread block where block may be
    described by a 1-, 2-, or 3-dimensional rectangular basis.
    The linear thread id is obtained by linearizing the 1-, 2- or 3-dimensional
    index. For instance, if the basis is denoted as (BX, BY, BZ) and the thread
    id is denoted by (tx, ty, tz), the linear thread id is:
      `linear_id = tx + ty * BX + tz * BX * BY)`.
    The linear thread id is fixed for the duration of a GPU kernel.
    
    This linear id mapping attribute indicates a different linearization relation
    is applied locally to a loop nest. 
    
    For instance, if the new basis is denoted as (LBX, LBY, LBZ) the thread id
    in the new basis is:
      `(linear_id mod LBX , (linear_id / LBX) mod * LBY, linear_id / (LBX * LBY))`.
    This reinterpretation is only fixe for the duration of a loop nest.
    
    It can be consumed by lowering to generate GPU code.
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


def GPUMemorySpaceMappingAttr : GPU_Attr<"GPUMemorySpaceMapping", "memory_space", [
  DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ] >  {
  let parameters = (ins
    EnumParameter<GPU_AddressSpaceEnum>:$address_space
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining memory hierarchy for GPU devices.

    GPU Memory has three memory space, global, workgroup, and private. The global memory
    is visible to all workitems and workgroups, the workgroup memory is only available for workitems
    within a workgroup, and private memory is only visible to a single workitem. This attribute indicates
    that using memory hiearchy is desired. It can be consumed by lowering to
    move data to a specific address space in GPU code.
  }];
}


#endif // GPU_DEVICE_MAPPING_ATTR
