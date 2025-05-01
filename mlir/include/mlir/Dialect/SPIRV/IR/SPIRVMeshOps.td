//===-- SPIRVMeshOps.td - MLIR SPIR-V Mesh Ops ------*- tablegen -*----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===------------------------------------------------------------------------------===//
//
// This file contains mesh ops for the SPIR-V dialect. It corresponds
// to the part of "3.52.25. Reserved Instructions" of the SPIR-V specification, and
// to the SPV_EXT_mesh_shader specification.
//
//===------------------------------------------------------------------------ -----===//

#ifndef MLIR_DIALECT_SPIRV_MESH_OPS
#define MLIR_DIALECT_SPIRV_MESH_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

// -----

def SPIRV_EXTEmitMeshTasksOp : SPIRV_ExtVendorOp<"EmitMeshTasks", [Terminator]> {
  let summary = [{
    Defines the grid size of subsequent mesh shader workgroups to generate upon
    completion of the task shader workgroup.
  }];

  let description = [{
    Defines the grid size of subsequent mesh shader workgroups to generate upon
    completion of the task shader workgroup.

    Group Count X Y Z must each be a 32-bit unsigned integer value. They
    configure the number of local workgroups in each respective dimensions for the
    launch of child mesh tasks. See Vulkan API specification for more detail.

    Payload is an optional pointer to the payload structure to pass to the
    generated mesh shader invocations. Payload must be the result of an OpVariable
    with a storage class of TaskPayloadWorkgroupEXT.

    The arguments are taken from the first invocation in each workgroup.
    Behaviour is undefined if any invocation terminates without executing this
    instruction, or if any invocation executes this instruction in non-uniform
    control flow.

    This instruction also serves as an OpControlBarrier instruction, and also
    performs and adheres to the description and semantics of an OpControlBarrier
    instruction with the Execution and Memory operands set to Workgroup and the
    Semantics operand set to a combination of WorkgroupMemory and AcquireRelease.

    Ceases all further processing: Only instructions executed before
    OpEmitMeshTasksEXT have observable side effects.

    This instruction must be the last instruction in a block.

    This instruction is only valid in the TaskEXT Execution Model.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    spirv.EmitMeshTasksEXT %x, %y, %z : i32, i32, i32
    spirv.EmitMeshTasksEXT %x, %x, %z, %payload : i32, i32, i32, !spirv.ptr<i32, TaskPayloadWorkgroupEXT>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_4>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_EXT_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingEXT]>
  ];

  let arguments = (ins
    SignlessOrUnsignedIntOfWidths<[32]>:$group_count_x,
    SignlessOrUnsignedIntOfWidths<[32]>:$group_count_y,
    SignlessOrUnsignedIntOfWidths<[32]>:$group_count_z,
    Optional<SPIRV_AnyPtr>:$payload
  );

  let results = (outs);

  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];
}

// -----

def SPIRV_EXTSetMeshOutputsOp : SPIRV_ExtVendorOp<"SetMeshOutputs", []> {
  let summary = [{
    Sets the actual output size of the primitives and vertices that the mesh
    shader workgroup will emit upon completion.
  }];

  let description = [{
    Vertex Count must be a 32-bit unsigned integer value. It defines the array size
    of per-vertex outputs.

    Primitive Count must a 32-bit unsigned integer value. It defines the array size
    of per-primitive outputs.

    The arguments are taken from the first invocation in each workgroup. Behavior
    is undefined if any invocation executes this instruction more than once or
    under non-uniform control flow. Behavior is undefined if there is any control
    flow path to an output write that is not preceded by this instruction.

    This instruction is only valid in the MeshEXT Execution Model.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    spirv.SetMeshOutputsEXT %vcount, %pcount : i32, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_4>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_EXT_mesh_shader]>,
    Capability<[SPIRV_C_MeshShadingEXT]>
  ];

  let arguments = (ins
    SignlessOrUnsignedIntOfWidths<[32]>:$vertex_count,
    SignlessOrUnsignedIntOfWidths<[32]>:$primitive_count
  );

  let results = (outs);
  let hasVerifier = 0;

  let assemblyFormat = [{
    operands attr-dict `:` type(operands)
  }];
}

#endif // MLIR_DIALECT_SPIRV_MESH_OPS
