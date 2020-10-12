//===-- SPIRVOps.td - MLIR SPIR-V Op Definitions Spec ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main operation definition specification file for SPIR-V
// operations.
//
//===----------------------------------------------------------------------===//

// Note that for each op in this file and the included files for specific op
// categories, we use a tool to automatically generate certain sections in its
// definition: basic structure, summary, description. So modifications to these
// sections will not be respected. Modifications to op traits, arguments,
// results, and sections after the results are retained. Besides, ops must be
// separated via the '// -----' marker.

#ifndef SPIRV_OPS
#define SPIRV_OPS

include "mlir/Dialect/SPIRV/SPIRVBase.td"
include "mlir/Dialect/SPIRV/SPIRVArithmeticOps.td"
include "mlir/Dialect/SPIRV/SPIRVAtomicOps.td"
include "mlir/Dialect/SPIRV/SPIRVBitOps.td"
include "mlir/Dialect/SPIRV/SPIRVCastOps.td"
include "mlir/Dialect/SPIRV/SPIRVCompositeOps.td"
include "mlir/Dialect/SPIRV/SPIRVControlFlowOps.td"
include "mlir/Dialect/SPIRV/SPIRVCooperativeMatrixOps.td"
include "mlir/Dialect/SPIRV/SPIRVGLSLOps.td"
include "mlir/Dialect/SPIRV/SPIRVGroupOps.td"
include "mlir/Dialect/SPIRV/SPIRVLogicalOps.td"
include "mlir/Dialect/SPIRV/SPIRVMatrixOps.td"
include "mlir/Dialect/SPIRV/SPIRVNonUniformOps.td"
include "mlir/Dialect/SPIRV/SPIRVOCLOps.td"
include "mlir/Dialect/SPIRV/SPIRVStructureOps.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// -----

def SPV_AccessChainOp : SPV_Op<"AccessChain", [NoSideEffect]> {
  let summary = [{
    Create a pointer into a composite object that can be used with OpLoad
    and OpStore.
  }];

  let description = [{
    Result Type must be an OpTypePointer. Its Type operand must be the type
    reached by walking the Base’s type hierarchy down to the last provided
    index in Indexes, and its Storage Class operand must be the same as the
    Storage Class of Base.

    Base must be a pointer, pointing to the base of a composite object.

    Indexes walk the type hierarchy to the desired depth, potentially down
    to scalar granularity. The first index in Indexes will select the top-
    level member/element/component/element of the base composite. All
    composite constituents use zero-based numbering, as described by their
    OpType… instruction. The second index will apply similarly to that
    result, and so on. Once any non-composite type is reached, there must be
    no remaining (unused) indexes.

     Each index in Indexes

    - must be a scalar integer type,

    - is treated as a signed count, and

    - must be an OpConstant when indexing into a structure.

    <!-- End of AutoGen section -->
    ```
    access-chain-op ::= ssa-id `=` `spv.AccessChain` ssa-use
                        `[` ssa-use (',' ssa-use)* `]`
                        `:` pointer-type
    ```

    #### Example:

    ```mlir
    %0 = "spv.constant"() { value = 1: i32} : () -> i32
    %1 = spv.Variable : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
    %2 = spv.AccessChain %1[%0] : !spv.ptr<!spv.struct<f32, !spv.array<4xf32>>, Function>
    %3 = spv.Load "Function" %2 ["Volatile"] : !spv.array<4xf32>
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$base_ptr,
    Variadic<SPV_Integer>:$indices
  );

  let results = (outs
    SPV_AnyPtr:$component_ptr
  );

  let builders = [OpBuilder<[{Value basePtr, ValueRange indices}]>];

  let hasCanonicalizer = 1;
}

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

def SPV_CopyMemoryOp : SPV_Op<"CopyMemory", []> {
  let summary = [{
    Copy from the memory pointed to by Source to the memory pointed to by
    Target. Both operands must be non-void pointers and having the same <id>
    Type operand in their OpTypePointer type declaration.  Matching Storage
    Class is not required.  The amount of memory copied is the size of the
    type pointed to. The copied type must have a fixed size; i.e., it cannot
    be, nor include, any OpTypeRuntimeArray types.
  }];

  let description = [{
    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory operand
    None. Before version 1.4, at most one memory operands mask can be
    provided. Starting with version 1.4 two masks can be provided, as
    described in Memory Operands. If no masks or only one mask is present,
    it applies to both Source and Target. If two masks are present, the
    first applies to Target and cannot include MakePointerVisible, and the
    second applies to Source and cannot include MakePointerAvailable.

    <!-- End of AutoGen section -->

    ```
    copy-memory-op ::= `spv.CopyMemory ` storage-class ssa-use
                       storage-class ssa-use
                       (`[` memory-access `]` (`, [` memory-access `]`)?)?
                       ` : ` spirv-element-type
    ```

    #### Example:

    ```mlir
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.Variable : !spv.ptr<f32, Function>
    spv.CopyMemory "Function" %0, "Function" %1 : f32
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$target,
    SPV_AnyPtr:$source,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment,
    OptionalAttr<SPV_MemoryAccessAttr>:$source_memory_access,
    OptionalAttr<I32Attr>:$source_alignment
  );

  let results = (outs);

  let verifier = [{ return verifyCopyMemory(*this); }];

  let autogenSerialization = 0;
}

// -----

def SPV_ExecutionModeOp : SPV_Op<"ExecutionMode", [InModuleScope]> {
  let summary = "Declare an execution mode for an entry point.";

  let description = [{
    Entry Point must be the Entry Point <id> operand of an OpEntryPoint
    instruction.

    Mode is the execution mode. See Execution Mode.

    This instruction is only valid when the Mode operand is an execution
    mode that takes no Extra Operands, or takes Extra Operands that are not
    <id> operands.

    <!-- End of AutoGen section -->

    ```
    execution-mode ::= "Invocations" | "SpacingEqual" |
                       <and other SPIR-V execution modes...>

    execution-mode-op ::= `spv.ExecutionMode ` ssa-use execution-mode
                          (integer-literal (`, ` integer-literal)* )?
    ```

    #### Example:

    ```mlir
    spv.ExecutionMode @foo "ContractionOff"
    spv.ExecutionMode @bar "LocalSizeHint", 3, 4, 5
    ```
  }];

  let arguments = (ins
    FlatSymbolRefAttr:$fn,
    SPV_ExecutionModeAttr:$execution_mode,
    I32ArrayAttr:$values
  );

  let results = (outs);

  let verifier = [{ return success(); }];

  let autogenSerialization = 0;

  let builders = [OpBuilder<[{spirv::FuncOp function,
                              spirv::ExecutionMode executionMode,
                              ArrayRef<int32_t> params}]>];
}

// -----

def SPV_LoadOp : SPV_Op<"Load", []> {
  let summary = "Load through a pointer.";

  let description = [{
    Result Type is the type of the loaded object. It must be a type with
    fixed size; i.e., it cannot be, nor include, any OpTypeRuntimeArray
    types.

    Pointer is the pointer to load through.  Its type must be an
    OpTypePointer whose Type operand is the same as Result Type.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory operand
    None.

    <!-- End of AutoGen section -->

    ```
    memory-access ::= `"None"` | `"Volatile"` | `"Aligned", ` integer-literal
                    | `"NonTemporal"`

    load-op ::= ssa-id ` = spv.Load ` storage-class ssa-use
                (`[` memory-access `]`)? ` : ` spirv-element-type
    ```

    #### Example:

    ```mlir
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.Load "Function" %0 : f32
    %2 = spv.Load "Function" %0 ["Volatile"] : f32
    %3 = spv.Load "Function" %0 ["Aligned", 4] : f32
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$ptr,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs
    SPV_Type:$value
  );

  let builders = [
    OpBuilder<[{
      Value basePtr, IntegerAttr memory_access = {},
      IntegerAttr alignment = {}
    }]>
  ];
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

// -----

def SPV_StoreOp : SPV_Op<"Store", []> {
  let summary = "Store through a pointer.";

  let description = [{
    Pointer is the pointer to store through.  Its type must be an
    OpTypePointer whose Type operand is the same as the type of Object.

    Object is the object to store.

    If present, any Memory Operands must begin with a memory operand
    literal. If not present, it is the same as specifying the memory operand
    None.

    <!-- End of AutoGen section -->

    ```
    store-op ::= `spv.Store ` storage-class ssa-use `, ` ssa-use `, `
                  (`[` memory-access `]`)? `:` spirv-element-type
    ```

    #### Example:

    ```mlir
    %0 = spv.Variable : !spv.ptr<f32, Function>
    %1 = spv.FMul ... : f32
    spv.Store "Function" %0, %1 : f32
    spv.Store "Function" %0, %1 ["Volatile"] : f32
    spv.Store "Function" %0, %1 ["Aligned", 4] : f32
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$ptr,
    SPV_Type:$value,
    OptionalAttr<SPV_MemoryAccessAttr>:$memory_access,
    OptionalAttr<I32Attr>:$alignment
  );

  let results = (outs);

  let builders = [
    OpBuilder<
      "Value ptr, Value value, ArrayRef<NamedAttribute> namedAttrs = {}", [{
      $_state.addOperands(ptr);
      $_state.addOperands(value);
      $_state.addAttributes(namedAttrs);
    }]>
  ];
}

// -----

def SPV_UndefOp : SPV_Op<"undef", []> {
  let summary = "Make an intermediate object whose value is undefined.";

  let description = [{
    Result Type is the type of object to make.

    Each consumption of Result <id> yields an arbitrary, possibly different
    bit pattern or abstract value resulting in possibly different concrete,
    abstract, or opaque values.

    <!-- End of AutoGen section -->

    ```
    undef-op ::= `spv.undef` `:` spirv-type
    ```

    #### Example:

    ```mlir
    %0 = spv.undef : f32
    %1 = spv.undef : !spv.struct<!spv.array<4 x vector<4xi32>>>
    ```
  }];

  let arguments = (ins);

  let results = (outs
    SPV_Type:$result
  );

  let verifier = [{ return success(); }];

  let hasOpcode = 0;
  let autogenSerialization = 0;

  let assemblyFormat = "attr-dict `:` type($result)";
}

// -----

def SPV_VariableOp : SPV_Op<"Variable", []> {
  let summary = [{
    Allocate an object in memory, resulting in a pointer to it, which can be
    used with OpLoad and OpStore.
  }];

  let description = [{
    Result Type must be an OpTypePointer. Its Type operand is the type of
    object in memory.

    Storage Class is the Storage Class of the memory holding the object.
    Since the op is used to model function-level variables, the storage class
    must be the `Function` Storage Class.

    Initializer is optional. If Initializer is present, it will be the
    initial value of the variable’s memory content. Initializer must be an
    <id> from a constant instruction or a global (module scope) OpVariable
    instruction. Initializer must have the same type as the type pointed to
    by Result Type.

    <!-- End of AutoGen section -->

    ```
    variable-op ::= ssa-id `=` `spv.Variable` (`init(` ssa-use `)`)?
                    attribute-dict? `:` spirv-pointer-type
    ```

    where `init` specifies initializer.

    #### Example:

    ```mlir
    %0 = spv.constant ...

    %1 = spv.Variable : !spv.ptr<f32, Function>
    %2 = spv.Variable init(%0): !spv.ptr<f32, Function>
    ```
  }];

  let arguments = (ins
    SPV_StorageClassAttr:$storage_class,
    Optional<AnyType>:$initializer
  );

  let results = (outs
    SPV_AnyPtr:$pointer
  );
}

// -----

#endif // SPIRV_OPS
