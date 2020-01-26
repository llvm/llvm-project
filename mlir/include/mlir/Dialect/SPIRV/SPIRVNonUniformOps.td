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

#ifndef SPIRV_NON_UNIFORM_OPS
#define SPIRV_NON_UNIFORM_OPS

// -----

def SPV_GroupNonUniformBallotOp : SPV_Op<"GroupNonUniformBallot", []> {
  let summary = [{
    Returns a bitfield value combining the Predicate value from all
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

    ### Custom assembly form

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    non-uniform-ballot-op ::= ssa-id `=` `spv.GroupNonUniformBallot` scope
                              ssa-use `:` `vector` `<` 4 `x` `integer-type` `>`
    ```

    For example:

    ```
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
    SPV_IntVec4:$result
  );
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

    ### Custom assembly form

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    non-uniform-elect-op ::= ssa-id `=` `spv.GroupNonUniformElect` scope
                             `:` `i1`
    ```

    For example:

    ```
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

  let builders = [
    OpBuilder<[{Builder *builder, OperationState &state, spirv::Scope}]>
  ];
}

// -----

def SPV_GroupNonUniformIAddOp : SPV_Op<"GroupNonUniformIAdd", []> {
  let summary = [{
    An integer add group operation of all Value operands contributed active
    by invocations in the group.
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

    ### Custom assembly form

    ```
    scope ::= `"Workgroup"` | `"Subgroup"`
    operation ::= `"Reduce"` | `"InclusiveScan"` | `"ExclusiveScan"` | ...
    integer-scalar-vector-type ::= integer-type |
                                 `vector<` integer-literal `x` integer-type `>`
    non-uniform-iadd-op ::= ssa-id `=` `spv.GroupNonUniformIAdd` scope operation
                            ssa-use ( `cluster_size` `(` ssa_use `)` )?
                            `:` integer-scalar-vector-type
    ```

    For example:

    ```
    %four = spv.constant 4 : i32
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

  let arguments = (ins
    SPV_ScopeAttr:$execution_scope,
    SPV_GroupOperationAttr:$group_operation,
    SPV_ScalarOrVectorOf<SPV_Integer>:$value,
    SPV_Optional<SPV_Integer>:$cluster_size
  );

  let results = (outs
    SPV_ScalarOrVectorOf<SPV_Integer>:$result
  );
}

// -----

#endif // SPIRV_NON_UNIFORM_OPS
