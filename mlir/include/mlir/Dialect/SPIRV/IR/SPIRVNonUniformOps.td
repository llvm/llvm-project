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

class SPV_GroupNonUniformArithmeticOp<string mnemonic, Type type,
      list<Trait> traits = []> : SPV_Op<mnemonic, traits> {

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope,
    SPV_GroupOperationAttr:$group_operation,
    SPV_ScalarOrVectorOf<type>:$value,
    Optional<SPV_Integer>:$cluster_size
  );

  let results = (outs
    SPV_ScalarOrVectorOf<type>:$result
  );

  let parser = [{ return parseGroupNonUniformArithmeticOp(parser, result); }];
  let printer = [{ printGroupNonUniformArithmeticOp(getOperation(), p); }];
}

// -----

def SPV_GroupNonUniformBallotOp : SPV_Op<"GroupNonUniformBallot", []> {
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

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    non-uniform-ballot-op ::= ssa-id `=` `spv.GroupNonUniformBallot` scope
                              ssa-use `:` `vector` `<` 4 `x` `integer-type` `>`
    ```

    #### Example:

    ```mlir
    %0 = spv.GroupNonUniformBallot "SubGroup" %predicate : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope,
    SPV_Bool:$predicate
  );

  let results = (outs
    SPV_IOrUIVec4:$result
  );

  let assemblyFormat = [{
    $execution_scope $predicate attr-dict `:` type($result)
  }];
}

// -----

def SPV_GroupNonUniformBroadcastOp : SPV_Op<"GroupNonUniformBroadcast",
  [NoSideEffect, AllTypesMatch<["value", "result"]>]> {
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

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    integer-float-scalar-vector-type ::= integer-type | float-type |
                               `vector<` integer-literal `x` integer-type `>` |
                               `vector<` integer-literal `x` float-type `>`
    group-non-uniform-broadcast-op ::= ssa-id `=` 
	            `spv.GroupNonUniformBroadcast` scope ssa_use,
                ssa_use `:` integer-float-scalar-vector-type `,` integer-type
    ```mlir

    #### Example:

    ```
    %scalar_value = ... : f32
    %vector_value = ... : vector<4xf32>
    %id = ... : i32
    %0 = spv.GroupNonUniformBroadcast "Subgroup" %scalar_value, %id : f32, i32
    %1 = spv.GroupNonUniformBroadcast "Workgroup" %vector_value, %id :
      vector<4xf32>, i32
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformBallot]>
  ];

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope,
    SPV_Type:$value,
    SPV_Integer:$id
  );

  let results = (outs
    SPV_Type:$result
  );

  let assemblyFormat = [{
    $execution_scope operands attr-dict `:` type($value) `,` type($id)
  }];
}

// -----

def SPV_GroupNonUniformElectOp : SPV_Op<"GroupNonUniformElect", []> {
  let summary = [{
    Result is true only in the active invocation with the lowest id in the
    group, otherwise result is false.
  }];

  let description = [{
    Result Type must be a Boolean type.

    Execution must be Workgroup or Subgroup Scope.

    <!-- End of AutoGen section -->

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    non-uniform-elect-op ::= ssa-id `=` `spv.GroupNonUniformElect` scope
                             `:` `i1`
    ```

    #### Example:

    ```mlir
    %0 = spv.GroupNonUniformElect : i1
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniform]>
  ];

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope
  );

  let results = (outs
    SPV_Bool:$result
  );

  let assemblyFormat = "$execution_scope attr-dict `:` type($result)";
}

// -----

def SPV_GroupNonUniformFAddOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformFAdd", SPV_Float, []> {
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    non-uniform-fadd-op ::= ssa-id `=` `spv.GroupNonUniformFAdd` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spv.GroupNonUniformFAdd "Workgroup" "Reduce" %scalar : f32
    %1 = spv.GroupNonUniformFAdd "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformFMaxOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformFMax", SPV_Float, []> {
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    non-uniform-fmax-op ::= ssa-id `=` `spv.GroupNonUniformFMax` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spv.GroupNonUniformFMax "Workgroup" "Reduce" %scalar : f32
    %1 = spv.GroupNonUniformFMax "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformFMinOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformFMin", SPV_Float, []> {
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    non-uniform-fmin-op ::= ssa-id `=` `spv.GroupNonUniformFMin` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spv.GroupNonUniformFMin "Workgroup" "Reduce" %scalar : f32
    %1 = spv.GroupNonUniformFMin "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformFMulOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformFMul", SPV_Float, []> {
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    float-scalar-vector-type ::= float-type |
                                 `vector<` integer-literal `x` float-type `>`
    non-uniform-fmul-op ::= ssa-id `=` `spv.GroupNonUniformFMul` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` float-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : f32
    %vector = ... : vector<4xf32>
    %0 = spv.GroupNonUniformFMul "Workgroup" "Reduce" %scalar : f32
    %1 = spv.GroupNonUniformFMul "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xf32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformIAddOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformIAdd", SPV_Integer, []> {
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-iadd-op ::= ssa-id `=` `spv.GroupNonUniformIAdd` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spv.GroupNonUniformIAdd "Workgroup" "Reduce" %scalar : i32
    %1 = spv.GroupNonUniformIAdd "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformIMulOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformIMul", SPV_Integer, []> {
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-imul-op ::= ssa-id `=` `spv.GroupNonUniformIMul` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spv.GroupNonUniformIMul "Workgroup" "Reduce" %scalar : i32
    %1 = spv.GroupNonUniformIMul "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformSMaxOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformSMax",
                                    SPV_Integer,
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-smax-op ::= ssa-id `=` `spv.GroupNonUniformSMax` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spv.GroupNonUniformSMax "Workgroup" "Reduce" %scalar : i32
    %1 = spv.GroupNonUniformSMax "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformSMinOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformSMin",
                                    SPV_Integer,
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-smin-op ::= ssa-id `=` `spv.GroupNonUniformSMin` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spv.GroupNonUniformSMin "Workgroup" "Reduce" %scalar : i32
    %1 = spv.GroupNonUniformSMin "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformUMaxOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformUMax",
                                    SPV_Integer,
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-umax-op ::= ssa-id `=` `spv.GroupNonUniformUMax` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spv.GroupNonUniformUMax "Workgroup" "Reduce" %scalar : i32
    %1 = spv.GroupNonUniformUMax "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

def SPV_GroupNonUniformUMinOp : SPV_GroupNonUniformArithmeticOp<"GroupNonUniformUMin",
                                    SPV_Integer,
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

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-umin-op ::= ssa-id `=` `spv.GroupNonUniformUMin` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    #### Example:

    ```mlir
    %four = spv.Constant 4 : i32
    %scalar = ... : i32
    %vector = ... : vector<4xi32>
    %0 = spv.GroupNonUniformUMin "Workgroup" "Reduce" %scalar : i32
    %1 = spv.GroupNonUniformUMin "Subgroup" "ClusteredReduce" %vector cluster_size(%four) : vector<4xi32>
    ```
  }];

  let availability = [
    MinVersion<SPV_V_1_3>,
    MaxVersion<SPV_V_1_5>,
    Extension<[]>,
    Capability<[SPV_C_GroupNonUniformArithmetic, SPV_C_GroupNonUniformClustered, SPV_C_GroupNonUniformPartitionedNV]>
  ];
}

// -----

#endif // MLIR_DIALECT_SPIRV_IR_NON_UNIFORM_OPS
