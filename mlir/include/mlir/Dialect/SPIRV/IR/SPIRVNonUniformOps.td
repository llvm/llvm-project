//===-- SPIRVNonUniformOps.td - MLIR SPIR-V NonUniform Ops -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains non-uniform ops for the SPIR-V dialect. It corresponds to
// "3.32.24. Non-Uniform Instructions" of the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPIRV_IR_NON_UNIFORM_OPS
#define MLIR_DIALECT_SPIRV_IR_NON_UNIFORM_OPS

class SPIRV_AttrIs<string operand, string type, string value> : PredOpTrait<
  operand # " must be " # type # " of value " # value,
  CPred<"::llvm::cast<::mlir::spirv::" # type # "Attr>(getProperties()." # operand # ").getValue() == ::mlir::spirv::" # type # "::" # value>
  >;

class SPIRV_ExecutionScopeAttrIs<string operand, string value> : SPIRV_AttrIs<operand, "Scope", value>;

// -----

class SPIRV_GroupNonUniformArithmeticOp<string mnemonic, Type type,
      list<Trait> traits = []> : SPIRV_Op<mnemonic, traits> {

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_ScalarOrVectorOf<type>:$value,
    Optional<SPIRV_Integer>:$cluster_size
  );

  let results = (outs
    SPIRV_ScalarOrVectorOf<type>:$result
  );  
  
  let hasCustomAssemblyFormat = 0;

  let assemblyFormat = [{
    $execution_scope $group_operation $value (`cluster_size``(` $cluster_size^ `)`)? attr-dict `:` type($value) (`,` type($cluster_size)^)? `->` type(results)
  }]; 
}

// -----

def SPIRV_GroupNonUniformBallotOp : SPIRV_Op<"GroupNonUniformBallot", []> {
  let summary = [{
    Result is a bitfield value combining the Predicate value from all
    invocations in the group that execute the same dynamic instance of this
    instruction. The bit is set to one if the corresponding invocation is
    active and the Predicate for that invocation evaluated to true;
    otherwise, it is set to zero.
  }];

  let description = [{
    Result Type  must be a vector of four components of integer type scalar,
    whose Signedness operand is 0.

    Result is a set of bitfields where the first invocation is represented
    in the lowest bit of the first vector component and the last (up to the
    size of the group) is the higher bit number of the last bitmask needed
    to represent all bits of the group invocations.

    Execution must be Workgroup or Subgroup Scope.

    Predicate must be a Boolean type.

    #### Example:

    ```mlir
    %0 = spirv.GroupNonUniformBallot <Subgroup> %predicate : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_Bool:$predicate
  );

  let results = (outs
    SPIRV_IOrUIVec4:$result
  );

  let assemblyFormat = [{
    $execution_scope $predicate attr-dict `:` type($result)
  }];
}

// -----

def SPIRV_GroupNonUniformBallotFindLSBOp : SPIRV_Op<"GroupNonUniformBallotFindLSB", []> {
  let summary = [{
    Find the least significant bit set to 1 in Value, considering only the
    bits in Value required to represent all bits of the group's invocations.
    If none of the considered bits is set to 1, the resulting value is
    undefined.
  }];

  let description = [{
    Result Type must be a scalar of integer type, whose Signedness operand
    is 0.

    Execution is a Scope that identifies the group of invocations affected
    by this command. It must be Subgroup.

    Value must be a vector of four components of integer type scalar, whose
    Width operand is 32 and whose Signedness operand is 0.

    Value is a set of bitfields where the first invocation is represented in
    the lowest bit of the first vector component and the last (up to the
    size of the group) is the higher bit number of the last bitmask needed
    to represent all bits of the group invocations.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformBallotFindLSB <Subgroup> %vector : vector<4xi32>, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_IOrUIVec4:$value
  );

  let results = (outs
    SPIRV_SignlessOrUnsignedInt:$result
  );

  let assemblyFormat = [{
    $execution_scope $value attr-dict `:` type($value) `,` type($result)
  }];
}

// -----

def SPIRV_GroupNonUniformBallotFindMSBOp : SPIRV_Op<"GroupNonUniformBallotFindMSB", []> {
  let summary = [{
    Find the most significant bit set to 1 in Value, considering only the
    bits in Value required to represent all bits of the group's invocations.
    If none of the considered bits is set to 1, the resulting value is
    undefined.
  }];

  let description = [{
    Result Type must be a scalar of integer type, whose Signedness operand
    is 0.

    Execution is a Scope that identifies the group of invocations affected
    by this command. It must be Subgroup.

    Value must be a vector of four components of integer type scalar, whose
    Width operand is 32 and whose Signedness operand is 0.

    Value is a set of bitfields where the first invocation is represented in
    the lowest bit of the first vector component and the last (up to the
    size of the group) is the higher bit number of the last bitmask needed
    to represent all bits of the group invocations.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformBallotFindMSB <Subgroup> %vector : vector<4xi32>, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_IOrUIVec4:$value
  );

  let results = (outs
    SPIRV_SignlessOrUnsignedInt:$result
  );

  let assemblyFormat = [{
    $execution_scope $value attr-dict `:` type($value) `,` type($result)
  }];
}

// -----

def SPIRV_GroupNonUniformBroadcastOp : SPIRV_Op<"GroupNonUniformBroadcast",
  [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Result is the Value of the invocation identified by the id Id to all
    active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution must be Workgroup or Subgroup Scope.

    The type of Value must be the same as Result Type.

    Id  must be a scalar of integer type, whose Signedness operand is 0.

    Before version 1.5, Id must come from a constant instruction. Starting
    with version 1.5, Id must be dynamically uniform.

    The resulting value is undefined if Id is an inactive invocation, or is
    greater than or equal to the size of the group.

    #### Example:

    ```mlir
    %scalar_value = ... : f32
    %vector_value = ... : vector<4xf32>
    %id = ... : i32
    %0 = spirv.GroupNonUniformBroadcast "Subgroup" %scalar_value, %id : f32, i32
    %1 = spirv.GroupNonUniformBroadcast "Workgroup" %vector_value, %id :
      vector<4xf32>, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_Type:$value,
    SPIRV_Integer:$id
  );

  let results = (outs
    SPIRV_Type:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($id)
  }];
}

// -----

def SPIRV_GroupNonUniformElectOp : SPIRV_Op<"GroupNonUniformElect", []> {
  let summary = [{
    Result is true only in the active invocation with the lowest id in the
    group, otherwise result is false.
  }];

  let description = [{
    Result Type must be a Boolean type.

    Execution must be Workgroup or Subgroup Scope.

    #### Example:

    ```mlir
    %0 = spirv.GroupNonUniformElect : i1
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniform]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope
  );

  let results = (outs
    SPIRV_Bool:$result
  );

  let assemblyFormat = "$execution_scope attr-dict `:` type($result)";
}

// -----

def SPIRV_GroupNonUniformFAddOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformFAdd", SPIRV_Float, []> {
  let summary = [{
    A floating point add group operation of all Value operands contributed
    by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.  The method used to
    perform the group operation on the contributed Value(s) from active
    invocations is implementation defined.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spirv.GroupNonUniformFAdd <Workgroup> <Reduce> %scalar : f32 -> f32
    %1 = spirv.GroupNonUniformFAdd <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformFMaxOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformFMax", SPIRV_Float, []> {
  let summary = [{
    A floating point maximum group operation of all Value operands
    contributed by active invocations in by group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is -INF. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.  The method used to
    perform the group operation on the contributed Value(s) from active
    invocations is implementation defined. From the set of Value(s) provided
    by active invocations within a subgroup, if for any two Values one of
    them is a NaN, the other is chosen. If all Value(s) that are used by the
    current invocation are NaN, then the result is an undefined value.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spirv.GroupNonUniformFMax <Workgroup> <Reduce> %scalar : f32 -> f32
    %1 = spirv.GroupNonUniformFMax <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformFMinOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformFMin", SPIRV_Float, []> {
  let summary = [{
    A floating point minimum group operation of all Value operands
    contributed by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is +INF. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.  The method used to
    perform the group operation on the contributed Value(s) from active
    invocations is implementation defined. From the set of Value(s) provided
    by active invocations within a subgroup, if for any two Values one of
    them is a NaN, the other is chosen. If all Value(s) that are used by the
    current invocation are NaN, then the result is an undefined value.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spirv.GroupNonUniformFMin <Workgroup> <Reduce> %scalar : f32 -> i32
    %1 = spirv.GroupNonUniformFMin <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformFMulOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformFMul", SPIRV_Float, []> {
  let summary = [{
    A floating point multiply group operation of all Value operands
    contributed by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is 1. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.  The method used to
    perform the group operation on the contributed Value(s) from active
    invocations is implementation defined.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spirv.GroupNonUniformFMul <Workgroup> <Reduce> %scalar : f32 -> f32
    %1 = spirv.GroupNonUniformFMul <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xf32>, i32 -> vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformIAddOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformIAdd", SPIRV_Integer, []> {
  let summary = [{
    An integer add group operation of all Value operands contributed by
    active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of integer type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformIAdd <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformIAdd <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformIMulOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformIMul", SPIRV_Integer, []> {
  let summary = [{
    An integer multiply group operation of all Value operands contributed by
    active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of integer type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is 1. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformIMul <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformIMul <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformSMaxOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformSMax",
                                    SPIRV_Integer,
                                    [SignedOp]> {
  let summary = [{
    A signed integer maximum group operation of all Value operands
    contributed by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of integer type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is INT_MIN. If Operation is
    ClusteredReduce, ClusterSize must be specified.

     The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformSMax <Workgroup> <Reduce> %scalar : i32
    %1 = spirv.GroupNonUniformSMax <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformSMinOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformSMin",
                                    SPIRV_Integer,
                                    [SignedOp]> {
  let summary = [{
    A signed integer minimum group operation of all Value operands
    contributed by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of integer type.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is INT_MAX. If Operation is
    ClusteredReduce, ClusterSize must be specified.

     The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformSMin <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformSMin <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformShuffleOp : SPIRV_Op<"GroupNonUniformShuffle",
    [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Result is the Value of the invocation identified by the id Id.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

     The type of Value must be the same as Result Type.

    Id  must be a scalar of integer type, whose Signedness operand is 0.

    The resulting value is undefined if Id is an inactive invocation, or is
    greater than or equal to the size of the group.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.GroupNonUniformShuffle <Subgroup> %val, %id : f32, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformShuffle]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_ScalarOrVector:$value,
    SPIRV_Integer:$id
  );

  let results = (outs
    SPIRV_ScalarOrVector:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($id)
  }];
}

// -----

def SPIRV_GroupNonUniformShuffleDownOp : SPIRV_Op<"GroupNonUniformShuffleDown",
    [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Result is the Value of the invocation identified by the current
    invocation’s id within the group + Delta.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

     The type of Value must be the same as Result Type.

    Delta  must be a scalar of integer type, whose Signedness operand is 0.

    Delta is treated as unsigned and the resulting value is undefined if
    Delta is greater than or equal to the size of the group, or if the
    current invocation’s id within the group + Delta is either an inactive
    invocation or greater than or equal to the size of the group.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.GroupNonUniformShuffleDown <Subgroup> %val, %delta : f32, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformShuffleRelative]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_ScalarOrVector:$value,
    SPIRV_Integer:$delta
  );

  let results = (outs
    SPIRV_Type:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($delta)
  }];
}

// -----

def SPIRV_GroupNonUniformShuffleUpOp : SPIRV_Op<"GroupNonUniformShuffleUp",
    [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Result is the Value of the invocation identified by the current
    invocation’s id within the group - Delta.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

     The type of Value must be the same as Result Type.

    Delta  must be a scalar of integer type, whose Signedness operand is 0.

    Delta is treated as unsigned and the resulting value is undefined if
    Delta is greater than the current invocation’s id within the group or if
    the selected lane is inactive.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.GroupNonUniformShuffleUp <Subgroup> %val, %delta : f32, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformShuffleRelative]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_ScalarOrVector:$value,
    SPIRV_Integer:$delta
  );

  let results = (outs
    SPIRV_Type:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($delta)
  }];
}

// -----

def SPIRV_GroupNonUniformShuffleXorOp : SPIRV_Op<"GroupNonUniformShuffleXor",
    [Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Result is the Value of the invocation identified by the current
    invocation’s id within the group xor’ed with Mask.
  }];

  let description = [{
    Result Type  must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

     The type of Value must be the same as Result Type.

    Mask  must be a scalar of integer type, whose Signedness operand is 0.

    The resulting value is undefined if current invocation’s id within the
    group xor’ed with Mask is an inactive invocation, or is greater than or
    equal to the size of the group.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %0 = spirv.GroupNonUniformShuffleXor <Subgroup> %val, %mask : f32, i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformShuffle]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_ScalarOrVector:$value,
    SPIRV_Integer:$mask
  );

  let results = (outs
    SPIRV_Type:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($mask)
  }];
}

// -----

def SPIRV_GroupNonUniformUMaxOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformUMax",
                                    SPIRV_Integer,
                                    [UnsignedOp]> {
  let summary = [{
    An unsigned integer maximum group operation of all Value operands
    contributed by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of integer type, whose
    Signedness operand is 0.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be specified.

     The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformUMax <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformUMax <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformUMinOp : SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformUMin",
                                    SPIRV_Integer,
                                    [UnsignedOp]> {
  let summary = [{
    An unsigned integer minimum group operation of all Value operands
    contributed by active invocations in the group.
  }];

  let description = [{
    Result Type  must be a scalar or vector of integer type, whose
    Signedness operand is 0.

    Execution must be Workgroup or Subgroup Scope.

    The identity I for Operation is UINT_MAX. If Operation is
    ClusteredReduce, ClusterSize must be specified.

     The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformUMin <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformUMin <Subgroup> <ClusteredReduce> %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic, SPIRV_C_GroupNonUniformClustered, SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformBitwiseAndOp :
  SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformBitwiseAnd",
    SPIRV_Integer, []> {
  let summary = [{
    A bitwise `and` group operation of all Value operands contributed by active
    invocations in the group.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is ~0. If Operation is ClusteredReduce,
    ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformBitwiseAnd <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformBitwiseAnd <Subgroup> <ClusteredReduce>
           %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic,
                SPIRV_C_GroupNonUniformClustered,
                SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformBitwiseOrOp :
  SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformBitwiseOr",
    SPIRV_Integer, []> {
  let summary = [{
    A bitwise `or` group operation of all Value operands contributed by active
    invocations in the group.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformBitwiseOr <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformBitwiseOr <Subgroup> <ClusteredReduce>
           %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic,
                SPIRV_C_GroupNonUniformClustered,
                SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformBitwiseXorOp :
  SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformBitwiseXor",
    SPIRV_Integer, []> {
  let summary = [{
    A bitwise `xor` group operation of all Value operands contributed by active
    invocations in the group.
  }];

  let description = [{
    Result Type must be a scalar or vector of integer type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spirv.GroupNonUniformBitwiseXor <Workgroup> <Reduce> %scalar : i32 -> i32
    %1 = spirv.GroupNonUniformBitwiseXor <Subgroup> <ClusteredReduce>
           %vector cluster_size(%four) : vector<4xi32>, i32 -> vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic,
                SPIRV_C_GroupNonUniformClustered,
                SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformLogicalAndOp :
  SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformLogicalAnd",
    SPIRV_Bool, []> {
  let summary = [{
    A logical `and` group operation of all Value operands contributed by active
    invocations in the group.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is ~0. If Operation is ClusteredReduce,
    ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i1
    %vector = ... : vector<4xi1>
    %0 = spirv.GroupNonUniformLogicalAnd <Workgroup> <Reduce> %scalar : i1 -> i1
    %1 = spirv.GroupNonUniformLogicalAnd <Subgroup> <ClusteredReduce>
           %vector cluster_size(%four) : vector<4xi1>, i32 -> vector<4xi1>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic,
                SPIRV_C_GroupNonUniformClustered,
                SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformLogicalOrOp :
  SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformLogicalOr",
    SPIRV_Bool, []> {
  let summary = [{
    A logical `or` group operation of all Value operands contributed by active
    invocations in the group.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i1
    %vector = ... : vector<4xi1>
    %0 = spirv.GroupNonUniformLogicalOr <Workgroup> <Reduce> %scalar : i1 -> i1
    %1 = spirv.GroupNonUniformLogicalOr <Subgroup> <ClusteredReduce>
           %vector cluster_size(%four) : vector<4xi1>, i32 -> vector<4xi1>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic,
                SPIRV_C_GroupNonUniformClustered,
                SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformLogicalXorOp :
  SPIRV_GroupNonUniformArithmeticOp<"GroupNonUniformLogicalXor",
    SPIRV_Bool, []> {
  let summary = [{
    A logical `xor` group operation of all Value operands contributed by active
    invocations in the group.
  }];

  let description = [{
    Result Type must be a scalar or vector of Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The identity I for Operation is 0. If Operation is ClusteredReduce,
    ClusterSize must be present.

    The type of Value must be the same as Result Type.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar
    of integer type, whose Signedness operand is 0. ClusterSize must come
    from a constant instruction. ClusterSize must be at least 1, and must be
    a power of 2. If ClusterSize is greater than the declared SubGroupSize,
    executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %scalar = ... : i1
    %vector = ... : vector<4xi1>
    %0 = spirv.GroupNonUniformLogicalXor <Workgroup> <Reduce> %scalar : i1 -> i1
    %1 = spirv.GroupNonUniformLogicalXor <Subgroup> <ClusteredReduce>
           %vector cluster_size(%four) : vector<4xi1>, i32 -> vector<4xi1>
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformArithmetic,
                SPIRV_C_GroupNonUniformClustered,
                SPIRV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPIRV_GroupNonUniformBallotBitCountOp : SPIRV_Op<"GroupNonUniformBallotBitCount", [
  SPIRV_ExecutionScopeAttrIs<"execution_scope", "Subgroup">,
]> {
  let summary = [{
    Result is the number of bits that are set to 1 in Value, considering
    only the bits in Value required to represent all bits of the scope
    restricted tangle.
  }];

  let description = [{
    Result Type must be a scalar of integer type, whose Signedness operand
    is 0.

    Execution is the scope defining the scope restricted tangle affected by
    this command. It must be Subgroup.

    The identity I for Operation is 0.

    Value must be a vector of four components of integer type scalar, whose
    Width operand is 32 and whose Signedness operand is 0.

    Value is a set of bitfields where the first invocation is represented in
    the lowest bit of the first vector component and the last (up to the
    size of the scope) is the higher bit number of the last bitmask needed
    to represent all bits of the invocations in the scope restricted tangle.

    An invocation will not execute a dynamic instance of this instruction
    (X') until all invocations in its scope restricted tangle have executed
    all dynamic instances that are program-ordered before X'.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %count = spirv.GroupNonUniformBallotBitCount <Subgroup> <Reduce> %val : vector<4xi32> -> i32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_GroupOperationAttr:$group_operation,
    SPIRV_Vec4<SignlessOrUnsignedIntOfWidths<[32]>>:$value
  );

  let results = (outs
    SPIRV_SignlessOrUnsignedInt:$result
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $execution_scope $group_operation $value attr-dict `:` type($value) `->` type($result)
  }];
}

// -----

def SPIRV_GroupNonUniformRotateKHROp : SPIRV_Op<"GroupNonUniformRotateKHR", [
  Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = [{
    Rotate values across invocations within a subgroup.
  }];

  let description = [{
    Return the Value of the invocation whose id within the group is calculated
    as follows:

    LocalId = SubgroupLocalInvocationId if Execution is Subgroup or
              LocalInvocationId if Execution is Workgroup
    RotationGroupSize = ClusterSize when ClusterSize is present, otherwise
    RotationGroupSize = SubgroupMaxSize if the Kernel capability is declared
                        and SubgroupSize if not.
    Invocation ID = ( (LocalId + Delta) & (RotationGroupSize - 1) ) +
                    (LocalId & ~(RotationGroupSize - 1))

    Result Type must be a scalar or vector of floating-point type, integer
    type, or Boolean type.

    Execution is a Scope. It must be either Workgroup or Subgroup.

    The type of Value must be the same as Result Type.

    Delta must be a scalar of integer type, whose Signedness operand is 0.
    Delta must be dynamically uniform within Execution.

    Delta is treated as unsigned and the resulting value is undefined if the
    selected lane is inactive.

    ClusterSize is the size of cluster to use. ClusterSize must be a scalar of
    integer type, whose Signedness operand is 0. ClusterSize must come from a
    constant instruction. Behavior is undefined unless ClusterSize is at least
    1 and a power of 2. If ClusterSize is greater than the declared
    SubGroupSize, executing this instruction results in undefined behavior.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %four = spirv.Constant 4 : i32
    %0 = spirv.GroupNonUniformRotateKHR <Subgroup> %value, %delta : f32, i32 -> f32
    %1 = spirv.GroupNonUniformRotateKHR <Workgroup> %value, %delta,
         clustersize(%four) : f32, i32, i32 -> f32
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformRotateKHR]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>, SPIRV_ScalarOrVectorOf<SPIRV_Bool>]>:$value,
    SPIRV_SignlessOrUnsignedInt:$delta,
    Optional<SPIRV_SignlessOrUnsignedInt>:$cluster_size
  );

  let results = (outs
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>, SPIRV_ScalarOrVectorOf<SPIRV_Bool>]>:$result
  );

  let assemblyFormat = [{
    $execution_scope $value `,` $delta (`,` `cluster_size` `(` $cluster_size^ `)`)? attr-dict `:` type($value) `,` type($delta) (`,` type($cluster_size)^)? `->` type(results)
  }];
}

// -----

def SPIRV_GroupNonUniformAllOp : SPIRV_Op<"GroupNonUniformAll", [
  SPIRV_ExecutionScopeAttrIs<"execution_scope", "Subgroup">
]> {
  let summary = [{
    Evaluates a predicate for all tangled invocations within the Execution
    scope, resulting in true if predicate evaluates to true for all tangled
    invocations within the Execution scope, otherwise the result is false.
  }];

  let description = [{
    Result Type must be a Boolean type.

    Execution is the scope defining the scope restricted tangle affected by
    this command. It must be Subgroup.

    Predicate must be a Boolean type.

    An invocation will not execute a dynamic instance of this instruction
    (X') until all invocations in its scope restricted tangle have executed
    all dynamic instances that are program-ordered before X'.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %predicate = ... : i1
    %0 = spirv.GroupNonUniformAll "Subgroup" %predicate : i1
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformVote]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_Bool:$predicate
  );

  let results = (outs
    SPIRV_Bool:$result
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $execution_scope $predicate attr-dict `:` type($result)
  }];
}

// -----

def SPIRV_GroupNonUniformAnyOp : SPIRV_Op<"GroupNonUniformAny", [
  SPIRV_ExecutionScopeAttrIs<"execution_scope", "Subgroup">
]> {
  let summary = [{
    Evaluates a predicate for all tangled invocations within the Execution
    scope, resulting in true if predicate evaluates to true for any tangled
    invocations within the Execution scope, otherwise the result is false.
  }];

  let description = [{
    Result Type must be a Boolean type.

    Execution is the scope defining the scope restricted tangle affected by
    this command. It must be Subgroup.

    Predicate must be a Boolean type.

    An invocation will not execute a dynamic instance of this instruction
    (X') until all invocations in its scope restricted tangle have executed
    all dynamic instances that are program-ordered before X'.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %predicate = ... : i1
    %0 = spirv.GroupNonUniformAny "Subgroup" %predicate : i1
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformVote]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    SPIRV_Bool:$predicate
  );

  let results = (outs
    SPIRV_Bool:$result
  );

  let hasVerifier = 0;

  let assemblyFormat = [{
    $execution_scope $predicate attr-dict `:` type($result)
  }];
}

// -----

def SPIRV_GroupNonUniformAllEqualOp : SPIRV_Op<"GroupNonUniformAllEqual", [
  SPIRV_ExecutionScopeAttrIs<"execution_scope", "Subgroup">
]> {
  let summary = [{
    Evaluates a value for all tangled invocations within the Execution
    scope. The result is true if Value is equal for all tangled invocations
    within the Execution scope. Otherwise, the result is false.
  }];

  let description = [{
    Result Type must be a Boolean type.

    Execution is the scope defining the scope restricted tangle affected by
    this command. It must be Subgroup.

    Value must be a scalar or vector of floating-point type, integer type,
    or Boolean type. The compare operation is based on this type, and if it
    is a floating-point type, an ordered-and-equal compare is used.

    An invocation will not execute a dynamic instance of this instruction
    (X') until all invocations in its scope restricted tangle have executed
    all dynamic instances that are program-ordered before X'.

    <!-- End of AutoGen section -->

    #### Example:

    ```mlir
    %scalar_value = ... : f32
    %vector_value = ... : vector<4xf32>
    %0 = spirv.GroupNonUniformAllEqual <Subgroup> %scalar_value : f32, i1
    %1 = spirv.GroupNonUniformAllEqual <Subgroup> %vector_value : vector<4xf32>, i1
    ```
  }];

  let availability = [
    MinVersion<SPIRV_V_1_3>,
    MaxVersion<SPIRV_V_1_6>,
    Extension<[]>,
    Capability<[SPIRV_C_GroupNonUniformVote]>
  ];

  let arguments = (ins
    SPIRV_ScopeAttr:$execution_scope,
    AnyTypeOf<[SPIRV_ScalarOrVectorOf<SPIRV_Float>, SPIRV_ScalarOrVectorOf<SPIRV_Integer>, SPIRV_ScalarOrVectorOf<SPIRV_Bool>]>:$value
  );

  let results = (outs
    SPIRV_Bool:$result
  );


  let hasVerifier = 0;

  let assemblyFormat = [{
    $execution_scope $value attr-dict `:` type($value) `,` type($result)
  }];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_NON_UNIFORM_OPS
