//===-- SPIRVMemoryOps.td - MLIR SPIR-V Memory Ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains memory ops for the SPIR-V dialect. It corresponds
// to "3.32.8. Memory Instructions" of the SPIR-V spec.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_MEMORY_OPS
#define MLIR_DIALECT_SPIRV_IR_MEMORY_OPS

include "mlir/Dialect/SPIRV/IR/SPIRVBase.td"

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
    %0 = "spv.Constant"() { value = 1: i32} : () -> i32
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

  let builders = [OpBuilder<(ins "Value":$basePtr, "ValueRange":$indices)>];

  let hasCanonicalizer = 1;
}

// -----

def SPV_CopyMemoryOp : SPV_Op<"CopyMemory", []> {
  let summary = [{
    Copy from the memory pointed to by Source to the memory pointed to by
    Target. Both operands must be non-void pointers and having the same <id>
    Type operand in their OpTypePointer type declaration.  Matching Storage
    Class is not required.  The amount of memory copied is the size of the
    type pointed to. The copied type must have a fixed size; i.e., it must
    not be, nor include, any OpTypeRuntimeArray types.
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
    OpBuilder<(ins "Value":$basePtr,
      CArg<"MemoryAccessAttr", "{}">:$memory_access,
      CArg<"IntegerAttr", "{}">:$alignment)>
  ];
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
    OpBuilder<(ins "Value":$ptr, "Value":$value,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$namedAttrs),
    [{
      $_state.addOperands(ptr);
      $_state.addOperands(value);
      $_state.addAttributes(namedAttrs);
    }]>
  ];
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
    %0 = spv.Constant ...

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

#endif // MLIR_DIALECT_SPIRV_IR_MEMORY_OPS
