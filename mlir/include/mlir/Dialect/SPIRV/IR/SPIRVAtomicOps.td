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

class SPIRV_AtomicUpdateOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_Op<mnemonic, traits> {
  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$semantics
  );

  let results = (outs
    SPIRV_Integer:$result
  );
}

class SPIRV_AtomicUpdateWithValueOp<string mnemonic, list<Trait> traits = []> :
  SPIRV_Op<mnemonic, traits> {
  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$semantics,
    SPIRV_Integer:$value
  );

  let results = (outs
    SPIRV_Integer:$result
  );

  let builders = [
    OpBuilder<(ins "Value":$pointer, "::mlir::spirv::Scope":$scope,
      "::mlir::spirv::MemorySemantics":$memory, "Value":$value),
    [{build($_builder, $_state, value.getType(), pointer, scope, memory, value);}]>
  ];
}

// -----

def SPIRV_AtomicAndOp : SPIRV_AtomicUpdateWithValueOp<"AtomicAnd", []> {
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
        `spirv.AtomicAnd` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicAnd "Device" "None" %pointer, %value :
                       !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicCompareExchangeOp : SPIRV_Op<"AtomicCompareExchange", []> {
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
        `spirv.AtomicCompareExchange` scope memory-semantics memory-semantics
                                    ssa-use `,` ssa-use `,` ssa-use
                                    `:` spv-pointer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.AtomicCompareExchange "Workgroup" "Acquire" "None"
                                    %pointer, %value, %comparator
                                    : !spirv.ptr<i32, WorkGroup>
    ```
  }];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$equal_semantics,
    SPIRV_MemorySemanticsAttr:$unequal_semantics,
    SPIRV_Integer:$value,
    SPIRV_Integer:$comparator
  );

  let results = (outs
    SPIRV_Integer:$result
  );
}

// -----

def SPIRV_AtomicCompareExchangeWeakOp : SPIRV_Op<"AtomicCompareExchangeWeak", []> {
  let summary = "Deprecated (use OpAtomicCompareExchange).";

  let description = [{
    Has the same semantics as OpAtomicCompareExchange.

    Memory must be a valid memory Scope.

    <!-- End of AutoGen section -->

    ```
    atomic-compare-exchange-weak-op ::=
        `spirv.AtomicCompareExchangeWeak` scope memory-semantics memory-semantics
                                        ssa-use `,` ssa-use `,` ssa-use
                                        `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicCompareExchangeWeak "Workgroup" "Acquire" "None"
                                       %pointer, %value, %comparator
                                       : !spirv.ptr<i32, WorkGroup>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_3>,
    Extension<[]>,
    Capability<[SPIRV_C_Kernel]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$equal_semantics,
    SPIRV_MemorySemanticsAttr:$unequal_semantics,
    SPIRV_Integer:$value,
    SPIRV_Integer:$comparator
  );

  let results = (outs
    SPIRV_Integer:$result
  );
}

// -----

def SPIRV_AtomicExchangeOp : SPIRV_Op<"AtomicExchange", []> {
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
        `spirv.AtomicCompareExchange` scope memory-semantics
                                    ssa-use `,` ssa-use `:` spv-pointer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.AtomicExchange "Workgroup" "Acquire" %pointer, %value,
                            : !spirv.ptr<i32, WorkGroup>
    ```
  }];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$semantics,
    SPIRV_Numerical:$value
  );

  let results = (outs
    SPIRV_Numerical:$result
  );
}

// -----

def SPIRV_EXTAtomicFAddOp : SPIRV_ExtVendorOp<"AtomicFAdd", []> {
  let summary = "TBD";

  let description = [{


    <!-- End of AutoGen section -->

    Perform the following steps atomically with respect to any other atomic
    accesses within Scope to the same location:

    1) load through Pointer to get an Original Value,

    2) get a New Value by float addition of Original Value and Value, and

    3) store the New Value back through Pointer.

    The instruction's result is the Original Value.

    Result Type must be a floating-point type scalar.

    The type of Value must be the same as Result Type. The type of the value
    pointed to by Pointer must be the same as Result Type.

    Memory must be a valid memory Scope.

    ```
    atomic-fadd-op ::=
        `spirv.EXT.AtomicFAdd` scope memory-semantics
                            ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.EXT.AtomicFAdd "Device" "None" %pointer, %value :
                           !spirv.ptr<f32, StorageBuffer>
    ```mlir
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_EXT_shader_atomic_float_add]>,
    Capability<[SPIRV_C_AtomicFloat16AddEXT, SPIRV_C_AtomicFloat32AddEXT, SPIRV_C_AtomicFloat64AddEXT]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$pointer,
    SPIRV_ScopeAttr:$memory_scope,
    SPIRV_MemorySemanticsAttr:$semantics,
    SPIRV_Float:$value
  );

  let results = (outs
    SPIRV_Float:$result
  );
}

// -----

def SPIRV_AtomicIAddOp : SPIRV_AtomicUpdateWithValueOp<"AtomicIAdd", []> {
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
        `spirv.AtomicIAdd` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicIAdd "Device" "None" %pointer, %value :
                        !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicIDecrementOp : SPIRV_AtomicUpdateOp<"AtomicIDecrement", []> {
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
        `spirv.AtomicIDecrement` scope memory-semantics ssa-use
                               `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicIDecrement "Device" "None" %pointer :
                              !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicIIncrementOp : SPIRV_AtomicUpdateOp<"AtomicIIncrement", []> {
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
        `spirv.AtomicIIncrement` scope memory-semantics ssa-use
                               `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicIncrement "Device" "None" %pointer :
                             !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicISubOp : SPIRV_AtomicUpdateWithValueOp<"AtomicISub", []> {
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
        `spirv.AtomicISub` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicISub "Device" "None" %pointer, %value :
                        !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicOrOp : SPIRV_AtomicUpdateWithValueOp<"AtomicOr", []> {
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
        `spirv.AtomicOr` scope memory-semantics
                       ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicOr "Device" "None" %pointer, %value :
                      !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicSMaxOp : SPIRV_AtomicUpdateWithValueOp<"AtomicSMax", []> {
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
        `spirv.AtomicSMax` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicSMax "Device" "None" %pointer, %value :
                        !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicSMinOp : SPIRV_AtomicUpdateWithValueOp<"AtomicSMin", []> {
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
        `spirv.AtomicSMin` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicSMin "Device" "None" %pointer, %value :
                        !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicUMaxOp : SPIRV_AtomicUpdateWithValueOp<"AtomicUMax", [UnsignedOp]> {
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
        `spirv.AtomicUMax` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicUMax "Device" "None" %pointer, %value :
                        !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicUMinOp : SPIRV_AtomicUpdateWithValueOp<"AtomicUMin", [UnsignedOp]> {
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
        `spirv.AtomicUMin` scope memory-semantics
                         ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicUMin "Device" "None" %pointer, %value :
                        !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

def SPIRV_AtomicXorOp : SPIRV_AtomicUpdateWithValueOp<"AtomicXor", []> {
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
        `spirv.AtomicXor` scope memory-semantics
                        ssa-use `,` ssa-use `:` spv-pointer-type
    ```

    #### Example:

    ```mlir
    %0 = spirv.AtomicXor "Device" "None" %pointer, %value :
                       !spirv.ptr<i32, StorageBuffer>
    ```
  }];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_ATOMIC_OPS
