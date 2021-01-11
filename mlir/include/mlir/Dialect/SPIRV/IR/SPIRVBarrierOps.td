//===-- SPIRVBarrierOps.td - MLIR SPIR-V Barrier Ops -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains barrier ops for the SPIR-V dialect. It corresponds
// to "3.32.20. Barrrier Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_BARRIER_OPS
#define MLIR_DIALECT_SPIRV_IR_BARRIER_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

// -----

def SPV_ControlBarrierOp : SPV_Op<"ControlBarrier", []> {
  let summary = [{
    Wait for other invocations of this module to reach the current point of
    execution.
  }];

  let description = [{
    All invocations of this module within Execution scope must reach this
    point of execution before any invocation will proceed beyond it.

    When Execution is Workgroup or larger, behavior is undefined if this
    instruction is used in control flow that is non-uniform within
    Execution. When Execution is Subgroup or Invocation, the behavior of
    this instruction in non-uniform control flow is defined by the client
    API.

    If Semantics is not None, this instruction also serves as an
    OpMemoryBarrier instruction, and must also perform and adhere to the
    description and semantics of an OpMemoryBarrier instruction with the
    same Memory and Semantics operands.  This allows atomically specifying
    both a control barrier and a memory barrier (that is, without needing
    two instructions). If Semantics is None, Memory is ignored.

    Before version 1.3, it is only valid to use this instruction with
    TessellationControl, GLCompute, or Kernel execution models. There is no
    such restriction starting with version 1.3.

    When used with the TessellationControl execution model, it also
    implicitly synchronizes the Output Storage Class:  Writes to Output
    variables performed by any invocation executed prior to a
    OpControlBarrier will be visible to any other invocation after return
    from that OpControlBarrier.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"CrossDevice"` | `"Device"` | `"Workgroup"` | ...

    memory-semantics ::= `"None"` | `"Acquire"` | "Release"` | ...

    control-barrier-op ::= `spv.ControlBarrier` scope, scope, memory-semantics
    ```

    #### Example:

    ```mlir
    spv.ControlBarrier "Workgroup", "Device", "Acquire|UniformMemory"

    ```
  }];

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$memory_semantics
  );

  let results = (outs);

  let verifier = [{ return verifyMemorySemantics(*this); }];

  let autogenSerialization = 0;

  let assemblyFormat = [{
    $execution_scope `,` $memory_scope `,` $memory_semantics attr-dict
  }];
}

// -----

def SPV_MemoryBarrierOp : SPV_Op<"MemoryBarrier", []> {
  let summary = "Control the order that memory accesses are observed.";

  let description = [{
    Ensures that memory accesses issued before this instruction will be
    observed before memory accesses issued after this instruction. This
    control is ensured only for memory accesses issued by this invocation
    and observed by another invocation executing within Memory scope. If the
    Vulkan memory model is declared, this ordering only applies to memory
    accesses that use the NonPrivatePointer memory operand or
    NonPrivateTexel image operand.

    Semantics declares what kind of memory is being controlled and what kind
    of control to apply.

    To execute both a memory barrier and a control barrier, see
    OpControlBarrier.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"CrossDevice"` | `"Device"` | `"Workgroup"` | ...

    memory-semantics ::= `"None"` | `"Acquire"` | `"Release"` | ...

    memory-barrier-op ::= `spv.MemoryBarrier` scope, memory-semantics
    ```

    #### Example:

    ```mlir
    spv.MemoryBarrier "Device", "Acquire|UniformMemory"

    ```
  }];

  let arguments = (ins
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$memory_semantics
  );

  let results = (outs);

  let verifier = [{ return verifyMemorySemantics(*this); }];

  let autogenSerialization = 0;

  let assemblyFormat = "$memory_scope `,` $memory_semantics attr-dict";
}

#endif // MLIR_DIALECT_SPIRV_IR_BARRIER_OPS
