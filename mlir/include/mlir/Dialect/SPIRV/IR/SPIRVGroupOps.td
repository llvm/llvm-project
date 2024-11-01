//===-- SPIRVGroupOps.td - MLIR SPIR-V (Sub)Group Ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains group and subgroup ops for the SPIR-V dialect. It
// corresponds to "3.32.21. Group and Subgroup Instructions" of the SPIR-V
// specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_GROUP_OPS
#define MLIR_DIALECT_SPIRV_IR_GROUP_OPS

// -----

def SPIRV_GroupFMulKHROp : SPIRV_KhrVendorOp<"GroupFMul", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    A floating-point multiplication group operation specified for all values of
    'X' specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    'Execution' reach this point of execution.

    Behavior is undefined unless all invocations within 'Execution' execute the
    same dynamic instance of this instruction.

    'Result Type' must be a scalar or vector of floating-point type.

    'Execution' is a Scope. It must be either Workgroup or Subgroup.

    The identity I for 'Operation' is 1.

    The type of 'X' must be the same as 'Result Type'.


    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.KHR.GroupFMul` scope operation ssa-use
                            `:` float-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.KHR.GroupFMul <Workgroup> <Reduce> %value : f32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupUniformArithmeticKHR]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupBroadcastOp : SPIRV_Op<"GroupBroadcast",
                              [Pure,
                               AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Broadcast the Value of the invocation identified by the local id LocalId
    to the result of all invocations in the group.
  }];

  let description = [{
    All invocations of this module within Execution must reach this point of
    execution.

    Behavior is undefined if this instruction is used in control flow that
    is non-uniform within Execution.

    Result Type  must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution must be Workgroup or Subgroup Scope.

     The type of Value must be the same as Result Type.

    LocalId must be an integer datatype. It can be a scalar, or a vector
    with 2 components or a vector with 3 components. LocalId must be the
    same for all invocations in the group.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    integer-float-scalar-vector-type ::= integer-type | float-type |
                               `vector<` integer-literal `x` integer-type `>` |
                               `vector<` integer-literal `x` float-type `>`
    localid-type ::= integer-type |
                   `vector<` integer-literal `x` integer-type `>`
    group-broadcast-op ::= ssa-id `=` `spirv.GroupBroadcast` scope ssa_use,
                   ssa_use `:` integer-float-scalar-vector-type `,` localid-type
    ```mlir

    #### Example:

    ```
    %scalar_value = ... : f32
    %vector_value = ... : vector<4xf32>
    %scalar_localid = ... : i32
    %vector_localid = ... : vector<3xi32>
    %0 = spirv.GroupBroadcast "Subgroup" %scalar_value, %scalar_localid : f32, i32
    %1 = spirv.GroupBroadcast "Workgroup" %vector_value, %vector_localid :
      vector<4xf32>, vector<3xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_Type:$value,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$localid
  );

  let results = (outs
    SPIRV_Type:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($localid)
  }];
}

// -----

def SPIRV_GroupFAddOp : SPIRV_Op<"GroupFAdd", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    A floating-point add group operation specified for all values of X
    specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of floating-point type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupFAdd` scope operation ssa-use
                            `:` float-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupFAdd <Workgroup> <Reduce> %value : f32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupFMaxOp : SPIRV_Op<"GroupFMax", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    A floating-point maximum group operation specified for all values of X
    specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of floating-point type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is -INF.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupFMax` scope operation ssa-use
                            `:` float-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupFMax <Workgroup> <Reduce> %value : f32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupFMinOp : SPIRV_Op<"GroupFMin", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    A floating-point minimum group operation specified for all values of X
    specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of floating-point type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is +INF.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupFMin` scope operation ssa-use
                            `:` float-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupFMin <Workgroup> <Reduce> %value : f32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Float>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupIAddOp : SPIRV_Op<"GroupIAdd", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    An integer add group operation specified for all values of X specified
    by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupIAdd` scope operation ssa-use
                            `:` integer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupIAdd <Workgroup> <Reduce> %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupIMulKHROp : SPIRV_KhrVendorOp<"GroupIMul", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    An integer multiplication group operation specified for all values of 'X'
    specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    'Execution' reach this point of execution.

    Behavior is undefined unless all invocations within 'Execution' execute the
    same dynamic instance of this instruction.

    'Result Type' must be a scalar or vector of integer type.

    'Execution' is a Scope. It must be either Workgroup or Subgroup.

    The identity I for 'Operation' is 1.

    The type of 'X' must be the same as 'Result Type'.


    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.KHR.GroupIMul` scope operation ssa-use
                            `:` integer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.KHR.GroupIMul <Workgroup> <Reduce> %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupUniformArithmeticKHR]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupSMaxOp : SPIRV_Op<"GroupSMax", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    A signed integer maximum group operation specified for all values of X
    specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is INT_MIN when X is 32 bits wide and
    LONG_MIN when X is 64 bits wide.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupSMax` scope operation ssa-use
                            `:` integer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupSMax <Workgroup> <Reduce> %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupSMinOp : SPIRV_Op<"GroupSMin", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    A signed integer minimum group operation specified for all values of X
    specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is INT_MAX when X is 32 bits wide and
    LONG_MAX when X is 64 bits wide.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupSMin` scope operation ssa-use
                            `:` integer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupSMin <Workgroup> <Reduce> %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupUMaxOp : SPIRV_Op<"GroupUMax", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    An unsigned integer maximum group operation specified for all values of
    X specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupUMax` scope operation ssa-use
                            `:` integer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupUMax <Workgroup> <Reduce> %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_GroupUMinOp : SPIRV_Op<"GroupUMin", [Pure,
                               AllTypesMatch<["x", "result"]>]> {
  let summary = [{
    An unsigned integer minimum group operation specified for all values of
    X specified by invocations in the group.
  }];

  let description = [{
    Behavior is undefined if not all invocations of this module within
    Execution reach this point of execution.

    Behavior is undefined unless all invocations within Execution execute
    the same dynamic instance of this instruction.

    Result Type  must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is UINT_MAX when X is 32 bits wide and
    ULONG_MAX when X is 64 bits wide.

     The type of X must be the same as Result Type.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"`
    op ::= ssa-id `=` `spirv.GroupUMin` scope operation ssa-use
                            `:` integer-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.GroupUMin <Workgroup> <Reduce> %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_Groups]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$x
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<SPIRV_Integer>:$result
  );

  let assemblyFormat = [{
    $execution_scope $group_operation operands attr-dict `:` type($x)
  }];
}

// -----

def SPIRV_INTELSubgroupBlockReadOp : SPIRV_IntelVendorOp<"SubgroupBlockRead", []> {
  let summary = "See extension SPV_INTEL_subgroups";

  let description = [{
    Reads one or more components of Result data for each invocation in the
    subgroup from the specified Ptr as a block operation.

    The data is read strided, so the first value read is:
    Ptr[ SubgroupLocalInvocationId ]

    and the second value read is:
    Ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ]
    etc.

    Result Type may be a scalar or vector type, and its component type must be
    equal to the type pointed to by Ptr.

    The type of Ptr must be a pointer type, and must point to a scalar type.

    <!-- End of AutoGen section -->

    ```
    subgroup-block-read-INTEL-op ::= ssa-id `=` `spirv.INTEL.SubgroupBlockRead`
                                storage-class ssa_use `:` spirv-element-type
    ```mlir

    #### Example:

    ```
    %0 = spirv.INTEL.SubgroupBlockRead "StorageBuffer" %ptr : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_subgroups]>,
    Capability<[SPIRV_C_SubgroupBufferBlockIOINTEL]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$ptr
  );

  let results = (outs
    SPIRV_Type:$value
  );
}

// -----

def SPIRV_INTELSubgroupBlockWriteOp : SPIRV_IntelVendorOp<"SubgroupBlockWrite", []> {
  let summary = "See extension SPV_INTEL_subgroups";

  let description = [{
    Writes one or more components of Data for each invocation in the subgroup
    from the specified Ptr as a block operation.

    The data is written strided, so the first value is written to:
    Ptr[ SubgroupLocalInvocationId ]

    and the second value written is:
    Ptr[ SubgroupLocalInvocationId + SubgroupMaxSize ]
    etc.

    The type of Ptr must be a pointer type, and must point to a scalar type.

    The component type of Data must be equal to the type pointed to by Ptr.

    <!-- End of AutoGen section -->

    ```
    subgroup-block-write-INTEL-op ::= ssa-id `=` `spirv.INTEL.SubgroupBlockWrite`
                      storage-class ssa_use `,` ssa-use `:` spirv-element-type
    ```mlir

    #### Example:

    ```
    spirv.INTEL.SubgroupBlockWrite "StorageBuffer" %ptr, %value : i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_INTEL_subgroups]>,
    Capability<[SPIRV_C_SubgroupBufferBlockIOINTEL]>
  ];

  let arguments = (ins
    SPIRV_AnyPtr:$ptr,
    SPIRV_Type:$value
  );

  let results = (outs);
}

// -----

def SPIRV_KHRSubgroupBallotOp : SPIRV_KhrVendorOp<"SubgroupBallot", []> {
  let summary = "See extension SPV_KHR_shader_ballot";

  let description = [{
    Computes a bitfield value combining the Predicate value from all invocations
    in the current Subgroup that execute the same dynamic instance of this
    instruction. The bit is set to one if the corresponding invocation is active
    and the predicate is evaluated to true; otherwise, it is set to zero.

    Predicate must be a Boolean type.

    Result Type must be a 4 component vector of 32 bit integer types.

    Result is a set of bitfields where the first invocation is represented in bit
    0 of the first vector component and the last (up to SubgroupSize) is the
    higher bit number of the last bitmask needed to represent all bits of the
    subgroup invocations.

    <!-- End of AutoGen section -->

    ```
    subgroup-ballot-op ::= ssa-id `=` `spirv.KHR.SubgroupBallot`
                                ssa-use `:` `vector` `<` 4 `x` `i32` `>`
    ```

    #### Example:

    ```mlir
    %0 = spirv.KHR.SubgroupBallot %predicate : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_0>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[SPV_KHR_shader_ballot]>,
    Capability<[SPIRV_C_SubgroupBallotKHR]>
  ];

  let arguments = (ins
    SPIRV_Bool:$predicate
  );

  let results = (outs
    SPIRV_Int32Vec4:$result
  );

  let hasVerifier = 0;

  let assemblyFormat = "$predicate attr-dict `:` type($result)";
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_GROUP_OPS
