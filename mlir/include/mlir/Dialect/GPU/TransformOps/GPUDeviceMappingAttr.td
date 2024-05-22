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
def LinearDim0 : I64EnumAttrCase<"LinearDim0", 3, "linear_dim_0">;
def LinearDim1 : I64EnumAttrCase<"LinearDim1", 4, "linear_dim_1">;
def LinearDim2 : I64EnumAttrCase<"LinearDim2", 5, "linear_dim_2">;
def LinearDim3 : I64EnumAttrCase<"LinearDim3", 6, "linear_dim_3">;
def LinearDim4 : I64EnumAttrCase<"LinearDim4", 7, "linear_dim_4">;
def LinearDim5 : I64EnumAttrCase<"LinearDim5", 8, "linear_dim_5">;
def LinearDim6 : I64EnumAttrCase<"LinearDim6", 9, "linear_dim_6">;
def LinearDim7 : I64EnumAttrCase<"LinearDim7", 10, "linear_dim_7">;
def LinearDim8 : I64EnumAttrCase<"LinearDim8", 11, "linear_dim_8">;
def LinearDim9 : I64EnumAttrCase<"LinearDim9", 12, "linear_dim_9">;

// TODO: This would be better represented with separate Grid and Linear Mapping
// ids. Unfortunately it is not yet possible to have an optional EnumParameter
// so we currently embed the 2 modes in the same enum.
def MappingIdEnum : I64EnumAttr<"MappingId", "Mapping ids for loop mapping", [
    DimX, DimY, DimZ,
    LinearDim0, LinearDim1, LinearDim2, LinearDim3, LinearDim4, 
    LinearDim5, LinearDim6, LinearDim7, LinearDim8, LinearDim9]> {
  let cppNamespace = "::mlir::gpu";
}

def GPUBlockMappingAttr : GPU_Attr<"GPUBlockMapping", "block", [
  DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ] >  {
  let parameters = (ins
    EnumParameter<MappingIdEnum>:$block
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining thread block parallelism for GPU devices.

    Thread blocks (aka workgroup) are grouped into a grid described by a 
    3-dimensional rectangle.
    This attribute indicates that thread block parallelism is desired.
    It can be consumed by lowering to generate GPU code.
    2 modes are supported: (1) 3D mapping mode and (2) linear mapping mode.

    #### 3D mapping mode

    The 3D block id is simply the 3D index of the block `(bidx, bidy, bidz)`. 
    If required, predication occurs on a per-dimension basis. This allows 
    specifying predication on a 3D sub-rectangle of the grid.

    #### Linear mapping mode

    The linear block id is obtained by linearizing the index of the block. 
    If required, predication occurs on the linear id. This allows specifying
    predication on a 1D subset of the (linearized) grid.

    For instance, if the basis is denoted as (GX, GY, GZ) and the block id is
    denoted by (bx, by, bz), the block id is:
      `linear_id = bx + by * GX + bz * GX * GBY)`.
    The linear block id is fixed for the duration of a GPU kernel.
    
    This linear id mapping attribute indicates a different linearization relation
    is applied locally to a loop nest. 
    
    For instance, if the new basis is denoted as (LBD0, LBD1, LBD2, LBD3) the 
    block id in the new basis is:
      ```(linear_id mod LBD0 , 
          (linear_id / LBD0) mod * LBD1, 
          (linear_id / (LBD0 * LBD1)) mod LBD2, 
          (linear_id / (LBD0 * LBD1 * LBD2)) mod LBD3)```.
    This reinterpretation is only fixed for the duration of a loop nest.
  }];
}

def GPUWarpgroupMappingAttr
    : GPU_Attr<"GPUWarpgroupMapping", "warpgroup", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ]> {
  let parameters = (ins
    EnumParameter<MappingIdEnum>:$warpgroup
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining warpgroup parallelism for GPU devices.

    Threads of proper granularity (e.g. multiple of 
    "kNumWarpsPerGroup * kWarpSize" on CUDA devices) can be grouped into
    warpgroups described by a 3-dimensional rectangle. 
    This attribute indicates that warpgroup parallelism is desired. 
    It can be consumed by lowering to generate GPU code.
    2 modes are supported: (1) 3D mapping mode and (2) linear mapping mode.

    #### 3D mapping mode

    The 3D warpgroup id is simply the adjusted 3D index of the thread 
    `(tidx / (kNumWarpsPerGroup * kWarpSize), tidy, tidz)`.
    If required, predication occurs on a per-dimension basis. This allows 
    specifying predication on a 3D sub-rectangle of the warpgroups.

    #### Linear mapping mode

    The linear warpgroup id is obtained by linearizing the index of the warpgroup.
    If required, predication occurs on the linear id. This allows specifying
    predication on a 1D "kNumWarpsPerGroup * kWarpSize"-aligned subset of the 
    (linearized) block.

    For instance, if the basis is denoted as (BX, BY, BZ) and the thread id is
    id is denoted by (tx, ty, tz), the linear warpgroup id is:
      ```linear_id = (tx + ty * BX + tz * BX * BY) 
                 / (kNumWarpsPerGroup * kWarpSize)```.
    The linear warpgroup id is fixed for the duration of a GPU kernel.
    
    This linear id mapping attribute indicates a different linearization relation
    is applied locally to a loop nest. 
    
    For instance, if the new basis is denoted as (LWGD0, LWGD1, LWGD2, LWGD3) the 
    warpgroup id in the new basis is:
      ```(linear_id mod LWGD0 , 
          (linear_id / LWGD0) mod * LWGD1, 
          (linear_id / (LWGD0 * LWGD1)) mod LWGD2, 
          (linear_id / (LWGD0 * LWGD1 * LWGD2)) mod LWGD3)```.
    This reinterpretation is only fixed for the duration of a loop nest.
  }];
}

def GPUWarpMappingAttr
    : GPU_Attr<"GPUWarpMapping", "warp", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ]> {
  let parameters = (ins
    EnumParameter<MappingIdEnum>:$warp
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining warp parallelism for GPU devices.

    Threads of proper granularity (e.g. multiple of "warp size" on CUDA devices) 
    can be grouped into warps described by a 3-dimensional rectangle. 
    This attribute indicates that warp parallelism is desired.
    It can be consumed by lowering to generate GPU code.
    2 modes are supported: (1) 3D mapping mode and (2) linear mapping mode.

    #### 3D mapping mode

    The 3D warp id is simply the adjusted 3D index of the thread 
    `(tidx / kWarpSize, tidy, tidz)`.
    If required, predication occurs on a per-dimension basis. This allows 
    specifying predication on a 3D sub-rectangle of the warpgroups.

    #### Linear mapping mode

    The linear warp id is obtained by linearizing the index of the warp.
    If required, predication occurs on the linear id. This allows specifying
    predication on a 1D "kWarpSize"-aligned subset of the (linearized) block.

    For instance, if the basis is denoted as (BX, BY, BZ) and the thread id is
    id is denoted by (tx, ty, tz), the linear warp id is:
      `linear_id = (tx + ty * BX + tz * BX * BY) / kWarpSize`.
    The linear warp id is fixed for the duration of a GPU kernel.
    
    This linear id mapping attribute indicates a different linearization relation
    is applied locally to a loop nest. 
    
    For instance, if the new basis is denoted as (LWD0, LWD1, LWD2, LWD3) the 
    warp id in the new basis is:
      ```(linear_id mod LWD0 , 
          (linear_id / LWD0) mod * LWD1, 
          (linear_id / (LWD0 * LWD1)) mod LWD2, 
          (linear_id / (LWD0 * LWD1 * LWD2)) mod LWD3)```.
    This reinterpretation is only fixed for the duration of a loop nest.
  }];
}

def GPUThreadMappingAttr
    : GPU_Attr<"GPUThreadMapping", "thread", [
      DeclareAttrInterfaceMethods<DeviceMappingAttrInterface> ]> {
  let parameters = (ins
    EnumParameter<MappingIdEnum>:$thread
  );
  let assemblyFormat = "`<` params `>`";
  let description = [{
    An attribute that allows defining thread parallelism for GPU devices.

    Thread (aka work item) are grouped into a thread blocks described by a 
    3-dimensional rectangle.
    This attribute indicates that thread parallelism is desired.
    It can be consumed by lowering to generate GPU.

    #### 3D mapping mode

    The 3D thread id is simply the 3D index of the thread `(tidx, tidy, tidz)`. 
    If required, predication occurs on a per-dimension basis. This allows 
    specifying predication on a 3D sub-rectangle of the block.

    #### Linear mapping mode

    The linear thread id is obtained by linearizing the index of the thread. 
    If required, predication occurs on the linear id. This allows specifying
    predication on a 1D subset of the (linearized) block.

    For instance, if the basis is denoted as (BX, BY, BZ) and the thread id is
    id is denoted by (tx, ty, tz), the linear thread id is:
      ```linear_id = (tx + ty * BX + tz * BX * BY)```.
    The linear thread id is fixed for the duration of a GPU kernel.
    
    This linear id mapping attribute indicates a different linearization relation
    is applied locally to a loop nest. 
    
    For instance, if the new basis is denoted as (LTD0, LTD1, LTD2, LTD3) the 
    thread id in the new basis is:
      ```(linear_id mod LTD0 , 
          (linear_id / LTD0) mod * LTD1, 
          (linear_id / (LTD0 * LTD1)) mod LTD2, 
          (linear_id / (LTD0 * LTD1 * LTD2)) mod LTD3)```.
    This reinterpretation is only fixed for the duration of a loop nest.
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
