//===-- SPIRVAtomicOps.td - MLIR SPIR-V Atomic Ops ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains atomic ops for the SPIR-V dialect. It corresponds to
// "3.32.18. Atomic Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_ATOMIC_OPS
#define MLIR_DIALECT_SPIRV_IR_ATOMIC_OPS

class SPV_AtomicUpdateOp<string mnemonic, list<Trait> traits = []> :
  SPV_Op<mnemonic, traits> {
  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$semantics
  );

  let results = (outs
    SPV_Integer:$result
  );
}

class SPV_AtomicUpdateWithValueOp<string mnemonic, list<Trait> traits = []> :
  SPV_Op<mnemonic, traits> {
  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$semantics,
    SPV_Integer:$value
  );

  let results = (outs
    SPV_Integer:$result
  );

  let builders = [
    OpBuilder<(ins "Value":$pointer, "::mlir::spirv::Scope":$scope,
      "::mlir::spirv::MemorySemantics":$memory, "Value":$value),
    [{build($_builder, $_state, value.getType(), pointer, scope, memory, value);}]>
  ];
}

// -----

def SPV_AtomicAndOp : SPV_AtomicUpdateWithValueOp<"AtomicAnd", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by the bitwise AND of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"CrossDevice"` | `"Device"` | `"Workgroup"` | ...

    memory-semantics ::= `"None"` | `"Acquire"` | "Release"` | ...

    atomic-and-op ::=
        `spv.AtomicAnd` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicAnd "Device" "None" %pointer, %value :
                       !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicCompareExchangeOp : SPV_Op<"AtomicCompareExchange", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value from Value only if Original Value equals Comparator,
    and

    3) store the New Value back through Pointer'only if 'Original Value
    equaled Comparator.

    The instruction's result is the Original Value.

    Result Type must be an integer type scalar.

    Use Equal for the memory semantics of this instruction when Value and
    Original Value compare equal.

    Use Unequal for the memory semantics of this instruction when Value and
    Original Value compare unequal. Unequal must not be set to Release or
    Acquire and Release. In addition, Unequal cannot be set to a stronger
    memory-order then Equal.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.  This type
    must also match the type of Comparator.

    Memory is a memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-compare-exchange-op ::=
        `spv.AtomicCompareExchange` scope memory-semantics memory-semantics
                                    ssa-use `,` ssa-use `,` ssa-use
                                    `:` spv-pointer-type
    ```mlir

    #### Example:

    ```
    %0 = spv.AtomicCompareExchange "Workgroup" "Acquire" "None"
                                    %pointer, %value, %comparator
                                    : !spv.ptr<i32, WorkGroup>
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$equal_semantics,
    SPV_MemorySemanticsAttr:$unequal_semantics,
    SPV_Integer:$value,
    SPV_Integer:$comparator
  );

  let results = (outs
    SPV_Integer:$result
  );
}

// -----

def SPV_AtomicCompareExchangeWeakOp : SPV_Op<"AtomicCompareExchangeWeak", []> {
  let summary = "Deprecated (use OpAtomicCompareExchange).";

  let description = [{
    Has the same semantics as OpAtomicCompareExchange.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-compare-exchange-weak-op ::=
        `spv.AtomicCompareExchangeWeak` scope memory-semantics memory-semantics
                                        ssa-use `,` ssa-use `,` ssa-use
                                        `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicCompareExchangeWeak "Workgroup" "Acquire" "None"
                                       %pointer, %value, %comparator
                                       : !spv.ptr<i32, WorkGroup>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_3>,
    Extension<[]>,
    Capability<[SPV_C_Kernel]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$equal_semantics,
    SPV_MemorySemanticsAttr:$unequal_semantics,
    SPV_Integer:$value,
    SPV_Integer:$comparator
  );

  let results = (outs
    SPV_Integer:$result
  );
}

// -----

def SPV_AtomicExchangeOp : SPV_Op<"AtomicExchange", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value from copying Value, and

    3) store the New Value back through Pointer.

    The instruction's result is the Original Value.

    Result Type must be a scalar of integer type or floating-point type.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory is a memory Scope.

    <!-- End of AutoGen section -->

     ```
    atomic-exchange-op ::=
        `spv.AtomicCompareExchange` scope memory-semantics
                                    ssa-use `,` ssa-use `:` spv-pointer-type
    ```mlir

    #### Example:

    ```
    %0 = spv.AtomicExchange "Workgroup" "Acquire" %pointer, %value,
                            : !spv.ptr<i32, WorkGroup>
    ```
  }];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$semantics,
    SPV_Numerical:$value
  );

  let results = (outs
    SPV_Numerical:$result
  );
}

// -----

def SPV_AtomicFAddEXTOp : SPV_Op<"AtomicFAddEXT", []> {
  let summary = "TBD";

  let description = [{


    <!-- End of AutoGen section -->

    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:

    1) load through Pointer to get an Original Value,

    2) get a New Value by float addition of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be a floating-point type scalar.

    The type of Value must be the same as Result Type. The type of the value
    pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ```
    atomic-fadd-op ::=
        `spv.AtomicFAddEXT` scope memory-semantics
                            ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicFAddEXT "Device" "None" %pointer, %value :
                           !spv.ptr<f32, StorageBuffer>
    ```mlir
  }];

  let availability = [
    MinVersion<SPV_V_1_0>,
    MaxVersion<SPV_V_1_6>,
    Extension<[SPV_EXT_shader_atomic_float_add]>,
    Capability<[SPV_C_AtomicFloat16AddEXT, SPV_C_AtomicFloat32AddEXT, SPV_C_AtomicFloat64AddEXT]>
  ];

  let arguments = (ins
    SPV_AnyPtr:$pointer,
    SPV_ScopeAttr:$memory_scope,
    SPV_MemorySemanticsAttr:$semantics,
    SPV_Float:$value
  );

  let results = (outs
    SPV_Float:$result
  );
}

// -----

def SPV_AtomicIAddOp : SPV_AtomicUpdateWithValueOp<"AtomicIAdd", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by integer addition of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-iadd-op ::=
        `spv.AtomicIAdd` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicIAdd "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicIDecrementOp : SPV_AtomicUpdateOp<"AtomicIDecrement", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value through integer subtraction of 1 from Original Value,
    and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.  The type of the value
    pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-idecrement-op ::=
        `spv.AtomicIDecrement` scope memory-semantics ssa-use
                               `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicIDecrement "Device" "None" %pointer :
                              !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicIIncrementOp : SPV_AtomicUpdateOp<"AtomicIIncrement", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value through integer addition of 1 to Original Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.  The type of the value
    pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-iincrement-op ::=
        `spv.AtomicIIncrement` scope memory-semantics ssa-use
                               `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicIncrement "Device" "None" %pointer :
                             !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicISubOp : SPV_AtomicUpdateWithValueOp<"AtomicISub", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by integer subtraction of Value from Original Value,
    and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-isub-op ::=
        `spv.AtomicISub` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicISub "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicOrOp : SPV_AtomicUpdateWithValueOp<"AtomicOr", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by the bitwise OR of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-or-op ::=
        `spv.AtomicOr` scope memory-semantics
                       ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicOr "Device" "None" %pointer, %value :
                      !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicSMaxOp : SPV_AtomicUpdateWithValueOp<"AtomicSMax", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the largest signed integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-smax-op ::=
        `spv.AtomicSMax` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicSMax "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicSMinOp : SPV_AtomicUpdateWithValueOp<"AtomicSMin", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the smallest signed integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-smin-op ::=
        `spv.AtomicSMin` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicSMin "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicUMaxOp : SPV_AtomicUpdateWithValueOp<"AtomicUMax", [UnsignedOp]> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the largest unsigned integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-umax-op ::=
        `spv.AtomicUMax` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicUMax "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicUMinOp : SPV_AtomicUpdateWithValueOp<"AtomicUMin", [UnsignedOp]> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by finding the smallest unsigned integer of Original
    Value and Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-umin-op ::=
        `spv.AtomicUMin` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicUMin "Device" "None" %pointer, %value :
                        !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPV_AtomicXorOp : SPV_AtomicUpdateWithValueOp<"AtomicXor", []> {
  let summary = [{
    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:
  }];

  let description = [{
    1) load through Pointer to get an Original Value,

    2) get a New Value by the bitwise exclusive OR of Original Value and
    Value, and

    3) store the New Value back through Pointer.

    The instruction’s result is the Original Value.

    Result Type must be an integer type scalar.

     The type of Value must be the same as Result Type.  The type of the
    value pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-xor-op ::=
        `spv.AtomicXor` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spv.AtomicXor "Device" "None" %pointer, %value :
                       !spv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_ATOMIC_OPS
