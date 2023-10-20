//===- SCFOps.td - Structured Control Flow operations ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines MLIR structured control flow operations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SCF_SCFOPS
#define MLIR_DIALECT_SCF_SCFOPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Dialect/SCF/IR/DeviceMappingInterface.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

def SCF_Dialect : Dialect {
  let name = "scf";
  let cppNamespace = "::mlir::scf";
  let dependentDialects = ["arith::ArithDialect"];
}

// Base class for SCF dialect ops.
class SCF_Op<string mnemonic, list<Trait> traits = []> :
    Op<SCF_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ConditionOp
//===----------------------------------------------------------------------===//

def ConditionOp : SCF_Op<"condition", [
  HasParent<"WhileOp">,
  DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
    ["getSuccessorRegions"]>,
  Pure,
  Terminator
]> {
  let summary = "loop continuation condition";
  let description = [{
    This operation accepts the continuation (i.e., inverse of exit) condition
    of the `scf.while` construct. If its first argument is true, the "after"
    region of `scf.while` is executed, with the remaining arguments forwarded
    to the entry block of the region. Otherwise, the loop terminates.
  }];

  let arguments = (ins I1:$condition, Variadic<AnyType>:$args);

  let assemblyFormat =
      [{ `(` $condition `)` attr-dict ($args^ `:` type($args))? }];
}

//===----------------------------------------------------------------------===//
// ExecuteRegionOp
//===----------------------------------------------------------------------===//

def ExecuteRegionOp : SCF_Op<"execute_region", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let summary = "operation that executes its region exactly once";
  let description = [{
    The `execute_region` operation is used to allow multiple blocks within SCF
    and other operations which can hold only one block.  The `execute_region`
    operation executes the region held exactly once and cannot have any operands.
    As such, its region has no arguments. All SSA values that dominate the op can
    be accessed inside the op. The op's region can have multiple blocks and the
    blocks can have multiple distinct terminators. Values returned from this op's
    region define the op's results.

    Example:

    ```mlir
    scf.for %i = 0 to 128 step %c1 {
      %y = scf.execute_region -> i32 {
        %x = load %A[%i] : memref<128xi32>
        scf.yield %x : i32
      }
    }

    affine.for %i = 0 to 100 {
      "foo"() : () -> ()
      %v = scf.execute_region -> i64 {
        cf.cond_br %cond, ^bb1, ^bb2

      ^bb1:
        %c1 = arith.constant 1 : i64
        cf.br ^bb3(%c1 : i64)

      ^bb2:
        %c2 = arith.constant 2 : i64
        cf.br ^bb3(%c2 : i64)

      ^bb3(%x : i64):
        scf.yield %x : i64
      }
      "bar"(%v) : (i64) -> ()
    }
    ```
  }];

  let results = (outs Variadic<AnyType>);

  let regions = (region AnyRegion:$region);

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ForOp
//===----------------------------------------------------------------------===//

def ForOp : SCF_Op<"for",
      [AutomaticAllocationScope, DeclareOpInterfaceMethods<LoopLikeOpInterface,
       ["getInits", "getSingleInductionVar", "getSingleLowerBound",
        "getSingleStep", "getSingleUpperBound", "getYieldedValues",
        "promoteIfSingleIteration", "replaceWithAdditionalYields"]>,
       AllTypesMatch<["lowerBound", "upperBound", "step"]>,
       ConditionallySpeculatable,
       DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands"]>,
       SingleBlockImplicitTerminator<"scf::YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "for operation";
  let description = [{
    The `scf.for` operation represents a loop taking 3 SSA value as operands
    that represent the lower bound, upper bound and step respectively. The
    operation defines an SSA value for its induction variable. It has one
    region capturing the loop body. The induction variable is represented as an
    argument of this region. This SSA value is a signless integer or index.
    The step is a value of same type but required to be positive. The lower and
    upper bounds specify a half-open range: the range includes the lower bound
    but does not include the upper bound.

    The body region must contain exactly one block that terminates with
    `scf.yield`. Calling ForOp::build will create such a region and insert
    the terminator implicitly if none is defined, so will the parsing even in
    cases when it is absent from the custom format. For example:

    ```mlir
    // Index case.
    scf.for %iv = %lb to %ub step %step {
      ... // body
    }
    ...
    // Integer case.
    scf.for %iv_32 = %lb_32 to %ub_32 step %step_32 : i32 {
      ... // body
    }
    ```

    `scf.for` can also operate on loop-carried variables and returns the final
    values after loop termination. The initial values of the variables are
    passed as additional SSA operands to the `scf.for` following the 3 loop
    control SSA values mentioned above (lower bound, upper bound and step). The
    operation region has an argument for the induction variable, followed by
    one argument for each loop-carried variable, representing the value of the
    variable at the current iteration.

    The region must terminate with a `scf.yield` that passes the current
    values of all loop-carried variables to the next iteration, or to the
    `scf.for` result, if at the last iteration. The static type of a
    loop-carried variable may not change with iterations; its runtime type is
    allowed to change. Note, that when the loop-carried variables are present,
    calling ForOp::build will not insert the terminator implicitly. The caller
    must insert `scf.yield` in that case.

    `scf.for` results hold the final values after the last iteration.
    For example, to sum-reduce a memref:

    ```mlir
    func.func @reduce(%buffer: memref<1024xf32>, %lb: index,
                      %ub: index, %step: index) -> (f32) {
      // Initial sum set to 0.
      %sum_0 = arith.constant 0.0 : f32
      // iter_args binds initial values to the loop's region arguments.
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %sum_next = arith.addf %sum_iter, %t : f32
        // Yield current iteration sum to next iteration %sum_iter or to %sum
        // if final iteration.
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```

    If the `scf.for` defines any values, a yield must be explicitly present.
    The number and types of the `scf.for` results must match the initial
    values in the `iter_args` binding and the yield operands.

    Another example with a nested `scf.if` (see `scf.if` for details) to
    perform conditional reduction:

    ```mlir
    func.func @conditional_reduce(%buffer: memref<1024xf32>, %lb: index,
                                  %ub: index, %step: index) -> (f32) {
      %sum_0 = arith.constant 0.0 : f32
      %c0 = arith.constant 0.0 : f32
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (f32) {
        %t = load %buffer[%iv] : memref<1024xf32>
        %cond = arith.cmpf "ugt", %t, %c0 : f32
        %sum_next = scf.if %cond -> (f32) {
          %new_sum = arith.addf %sum_iter, %t : f32
          scf.yield %new_sum : f32
        } else {
          scf.yield %sum_iter : f32
        }
        scf.yield %sum_next : f32
      }
      return %sum : f32
    }
    ```
  }];
  let arguments = (ins AnySignlessIntegerOrIndex:$lowerBound,
                       AnySignlessIntegerOrIndex:$upperBound,
                       AnySignlessIntegerOrIndex:$step,
                       Variadic<AnyType>:$initArgs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$lowerBound, "Value":$upperBound, "Value":$step,
      CArg<"ValueRange", "std::nullopt">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, ValueRange)>",
           "nullptr">)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, Value, ValueRange)>;

    Value getInductionVar() { return getBody()->getArgument(0); }

    Block::BlockArgListType getRegionIterArgs() {
      return getBody()->getArguments().drop_front(getNumInductionVars());
    }

    /// Return the `index`-th region iteration argument.
    BlockArgument getRegionIterArg(unsigned index) {
      assert(index < getNumRegionIterArgs() &&
        "expected an index less than the number of region iter args");
      return getBody()->getArguments().drop_front(getNumInductionVars())[index];
    }

    void setLowerBound(Value bound) { getOperation()->setOperand(0, bound); }
    void setUpperBound(Value bound) { getOperation()->setOperand(1, bound); }
    void setStep(Value step) { getOperation()->setOperand(2, step); }

    /// Number of induction variables, always 1 for scf::ForOp.
    unsigned getNumInductionVars() { return 1; }
    /// Number of region arguments for loop-carried values
    unsigned getNumRegionIterArgs() {
      return getBody()->getNumArguments() - getNumInductionVars();
    }
    /// Number of operands controlling the loop: lb, ub, step
    unsigned getNumControlOperands() { return 3; }

    /// Get the region iter arg that corresponds to an OpOperand.
    /// This helper prevents internal op implementation detail leakage to
    /// clients by hiding the operand / block argument mapping.
    BlockArgument getRegionIterArgForOpOperand(OpOperand &opOperand) {
      assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
             "expected an iter args operand");
      assert(opOperand.getOwner() == getOperation() &&
             "opOperand does not belong to this scf::ForOp operation");
      return getRegionIterArgs()[
        opOperand.getOperandNumber() - getNumControlOperands()];
    }
    /// Get the OpOperand& that corresponds to a region iter arg.
    /// This helper prevents internal op implementation detail leakage to
    /// clients by hiding the operand / block argument mapping.
    OpOperand &getOpOperandForRegionIterArg(BlockArgument bbArg) {
      assert(bbArg.getArgNumber() >= getNumInductionVars() &&
             "expected a bbArg that is not an induction variable");
      assert(bbArg.getOwner()->getParentOp() == getOperation() &&
             "bbArg does not belong to the scf::ForOp body");
      return getOperation()->getOpOperand(
        getNumControlOperands() + bbArg.getArgNumber() - getNumInductionVars());
    }
    /// Get the OpResult that corresponds to an OpOperand.
    /// Assert that opOperand is an iterArg.
    /// This helper prevents internal op implementation detail leakage to
    /// clients by hiding the operand / block argument mapping.
    OpResult getResultForOpOperand(OpOperand &opOperand) {
      assert(opOperand.getOperandNumber() >= getNumControlOperands() &&
             "expected an iter args operand");
      assert(opOperand.getOwner() == getOperation() &&
             "opOperand does not belong to this scf::ForOp operation");
      return getOperation()->getResult(
        opOperand.getOperandNumber() - getNumControlOperands());
    }
    /// Get the OpOperand& that corresponds to an OpResultOpOperand.
    /// This helper prevents internal op implementation detail leakage to
    /// clients by hiding the operand / block argument mapping.
    OpOperand &getOpOperandForResult(OpResult opResult) {
      assert(opResult.getDefiningOp() == getOperation() &&
             "opResult does not belong to the scf::ForOp operation");
      return getOperation()->getOpOperand(
        getNumControlOperands() + opResult.getResultNumber());
    }

    /// Returns the step as an `APInt` if it is constant.
    std::optional<APInt> getConstantStep();

    /// Interface method for ConditionallySpeculatable.
    Speculation::Speculatability getSpeculatability();
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ForallOp
//===----------------------------------------------------------------------===//

def ForallOp : SCF_Op<"forall", [
       AttrSizedOperandSegments,
       AutomaticAllocationScope,
       DeclareOpInterfaceMethods<LoopLikeOpInterface,
          ["promoteIfSingleIteration", "getSingleInductionVar",
          "getSingleLowerBound", "getSingleUpperBound", "getSingleStep"]>,
       RecursiveMemoryEffects,
       SingleBlockImplicitTerminator<"scf::InParallelOp">,
       DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       DestinationStyleOpInterface
     ]> {
  let summary = "evaluate a block multiple times in parallel";
  let description = [{
    `scf.forall` is a target-independent multi-dimensional parallel
    region application operation. It has exactly one block that represents the
    parallel body and it takes index operands that specify lower bounds, upper
    bounds and steps.

    The op also takes a variadic number of tensor operands (`shared_outs`).
    The future buffers corresponding to these tensors are shared among all
    threads. Shared tensors should be accessed via their corresponding block
    arguments. If multiple threads write to a shared buffer in a racy
    fashion, these writes will execute in some unspecified order. Tensors that
    are not shared can be used inside the body (i.e., the op is not isolated
    from above); however, if a use of such a tensor bufferizes to a memory
    write, the tensor is privatized, i.e., a thread-local copy of the tensor is
    used. This ensures that memory side effects of a thread are not visible to
    other threads (or in the parent body), apart from explicitly shared tensors.

    The name "thread" conveys the fact that the parallel execution is mapped
    (i.e. distributed) to a set of virtual threads of execution, one function
    application per thread. Further lowerings are responsible for specifying
    how this is materialized on concrete hardware resources.

    An optional `mapping` is an attribute array that specifies processing units
    with their dimension, how it remaps 1-1 to a set of concrete processing
    element resources (e.g. a CUDA grid dimension or a level of concrete nested
    async parallelism). It is expressed via any attribute that implements the
    device mapping interface. It is the reponsibility of the lowering mechanism
    to interpret the `mapping` attributes in the context of the concrete target
    the op is lowered to, or to ignore it when the specification is ill-formed
    or unsupported for a particular target.

    The only allowed terminator is `scf.forall.in_parallel`.
    `scf.forall` returns one value per `shared_out` operand. The
    actions of the `in_parallel` terminators specify how to combine the
    partial results of all parallel invocations into a full value, in some
    unspecified order. The "destination" of each such op must be a `shared_out`
    block argument of the `scf.forall` op.

    The actions involved in constructing the return values are further described
    by `tensor.parallel_insert_slice`.

    `scf.forall` acts as an implicit synchronization point.

    When the parallel function body has side effects, their order is unspecified
    across threads.

    `scf.forall` can be printed in two different ways depending on
    whether the loop is normalized or not. The loop is 'normalized' when all
    lower bounds are equal to zero and steps are equal to one. In that case,
    `lowerBound` and `step` operands will be omitted during printing.

    Normalized loop example:

    ```mlir
    //
    // Sequential context.
    //
    %matmul_and_pointwise:2 = scf.forall (%thread_id_1, %thread_id_2) in
        (%num_threads_1, %numthread_id_2) shared_outs(%o1 = %C, %o2 = %pointwise)
      -> (tensor<?x?xT>, tensor<?xT>) {
      //
      // Parallel context, each thread with id = (%thread_id_1, %thread_id_2)
      // runs its version of the code.
      //
      %sA = tensor.extract_slice %A[f((%thread_id_1, %thread_id_2))]:
        tensor<?x?xT> to tensor<?x?xT>
      %sB = tensor.extract_slice %B[g((%thread_id_1, %thread_id_2))]:
        tensor<?x?xT> to tensor<?x?xT>
      %sC = tensor.extract_slice %o1[h((%thread_id_1, %thread_id_2))]:
        tensor<?x?xT> to tensor<?x?xT>
      %sD = matmul ins(%sA, %sB) outs(%sC)

      %spointwise = subtensor %o2[i((%thread_id_1, %thread_id_2))]:
        tensor<?xT> to tensor<?xT>
      %sE = add ins(%spointwise) outs(%sD)

      scf.forall.in_parallel {
        scf.forall.parallel_insert_slice %sD into %o1[h((%thread_id_1, %thread_id_2))]:
          tensor<?x?xT> into tensor<?x?xT>

        scf.forall.parallel_insert_slice %spointwise into %o2[i((%thread_id_1, %thread_id_2))]:
          tensor<?xT> into tensor<?xT>
      }
    }
    // Implicit synchronization point.
    // Sequential context.
    //
    ```

    Loop with loop bounds example:

    ```mlir
    //
    // Sequential context.
    //
    %pointwise = scf.forall (%i, %j) = (0, 0) to (%dim1, %dim2)
      step (%tileSize1, %tileSize2) shared_outs(%o1 = %out)
      -> (tensor<?x?xT>, tensor<?xT>) {
      //
      // Parallel context.
      //
      %sA = tensor.extract_slice %A[%i, %j][%tileSize1, %tileSize2][1, 1]
        : tensor<?x?xT> to tensor<?x?xT>
      %sB = tensor.extract_slice %B[%i, %j][%tileSize1, %tileSize2][1, 1]
        : tensor<?x?xT> to tensor<?x?xT>
      %sC = tensor.extract_slice %o[%i, %j][%tileSize1, %tileSize2][1, 1]
        : tensor<?x?xT> to tensor<?x?xT>

      %add = map {"arith.addf"} ins(%sA, %sB) outs(%sC)

      scf.forall.in_parallel {
        scf.forall.parallel_insert_slice %add into
          %o[%i, %j][%tileSize1, %tileSize2][1, 1]
          : tensor<?x?xT> into tensor<?x?xT>
      }
    }
    // Implicit synchronization point.
    // Sequential context.
    //
    ```

    Example with mapping attribute:

    ```mlir
    //
    // Sequential context. Here `mapping` is expressed as GPU thread mapping
    // attributes
    //
    %matmul_and_pointwise:2 = scf.forall (%thread_id_1, %thread_id_2) in
        (%num_threads_1, %numthread_id_2) shared_outs(...)
      -> (tensor<?x?xT>, tensor<?xT>) {
      //
      // Parallel context, each thread with id = **(%thread_id_2, %thread_id_1)**
      // runs its version of the code.
      //
       scf.forall.in_parallel {
         ...
      }
    } { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    // Implicit synchronization point.
    // Sequential context.
    //
    ```

    Example with privatized tensors:

    ```mlir
    %t0 = ...
    %t1 = ...
    %r = scf.forall ... shared_outs(%o = t0) -> tensor<?xf32> {
      // %t0 and %t1 are privatized. %t0 is definitely copied for each thread
      // because the scf.forall op's %t0 use bufferizes to a memory
      // write. In the absence of other conflicts, %t1 is copied only if there
      // are uses of %t1 in the body that bufferize to a memory read and to a
      // memory write.
      "some_use"(%t0)
      "some_use"(%t1)
    }
    ```
  }];
  let arguments = (ins
    Variadic<Index>:$dynamicLowerBound,
    Variadic<Index>:$dynamicUpperBound,
    Variadic<Index>:$dynamicStep,
    DenseI64ArrayAttr:$staticLowerBound,
    DenseI64ArrayAttr:$staticUpperBound,
    DenseI64ArrayAttr:$staticStep,
    Variadic<AnyRankedTensor>:$outputs,
    OptionalAttr<DeviceMappingArrayAttr>:$mapping);

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
    // Builder that takes loop bounds.
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$lbs,
       "ArrayRef<OpFoldResult>":$ubs, "ArrayRef<OpFoldResult>":$steps,
       "ValueRange":$outputs, "std::optional<ArrayAttr>":$mapping,
       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
            "nullptr"> :$bodyBuilderFn)>,

    // Builder for normalized loop that takes only upper bounds.
    OpBuilder<(ins "ArrayRef<OpFoldResult>":$ubs,
       "ValueRange":$outputs, "std::optional<ArrayAttr>":$mapping,
       CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>",
            "nullptr"> :$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    // Get lower bounds as OpFoldResult.
    SmallVector<OpFoldResult> getMixedLowerBound() {
      Builder b(getOperation()->getContext());
      return getMixedValues(getStaticLowerBound(), getDynamicLowerBound(), b);
    }

    // Get upper bounds as OpFoldResult.
    SmallVector<OpFoldResult> getMixedUpperBound() {
      Builder b(getOperation()->getContext());
      return getMixedValues(getStaticUpperBound(), getDynamicUpperBound(), b);
    }

    // Get steps as OpFoldResult.
    SmallVector<OpFoldResult> getMixedStep() {
      Builder b(getOperation()->getContext());
      return getMixedValues(getStaticStep(), getDynamicStep(), b);
    }

    /// Get lower bounds as values.
    SmallVector<Value> getLowerBound(OpBuilder &b) {
      return getValueOrCreateConstantIndexOp(b, getLoc(), getMixedLowerBound());
    }

    /// Get upper bounds as values.
    SmallVector<Value> getUpperBound(OpBuilder &b) {
      return getValueOrCreateConstantIndexOp(b, getLoc(), getMixedUpperBound());
    }

    /// Get steps as values.
    SmallVector<Value> getStep(OpBuilder &b) {
      return getValueOrCreateConstantIndexOp(b, getLoc(), getMixedStep());
    }

    int64_t getRank() { return getStaticLowerBound().size(); }

    /// Number of operands controlling the loop: lbs, ubs, steps
    unsigned getNumControlOperands() { return 3 * getRank(); }

    /// Number of dynamic operands controlling the loop: lbs, ubs, steps
    unsigned getNumDynamicControlOperands() {
      return getODSOperandIndexAndLength(3).first;
    }

    OpResult getTiedOpResult(OpOperand *opOperand) {
      assert(opOperand->getOperandNumber() >= getNumDynamicControlOperands() &&
             "invalid operand");
      return getOperation()->getOpResult(
          opOperand->getOperandNumber() - getNumDynamicControlOperands());
    }

    /// Return the num_threads operand that is tied to the given thread id
    /// block argument.
    OpOperand *getTiedOpOperand(BlockArgument bbArg) {
      assert(bbArg.getArgNumber() >= getRank() && "invalid bbArg");

      return &getOperation()->getOpOperand(getNumDynamicControlOperands() +
                                           bbArg.getArgNumber() - getRank());
    }

    /// Return the shared_outs operand that is tied to the given OpResult.
    OpOperand *getTiedOpOperand(OpResult opResult) {
      assert(opResult.getDefiningOp() == getOperation() && "invalid OpResult");
      return &getOperation()->getOpOperand(getNumDynamicControlOperands() +
                                           opResult.getResultNumber());
    }

    BlockArgument getTiedBlockArgument(OpOperand *opOperand) {
      assert(opOperand->getOperandNumber() >= getNumDynamicControlOperands() &&
             "invalid operand");

      return getBody()->getArgument(opOperand->getOperandNumber() -
                                    getNumDynamicControlOperands() + getRank());
    }

    ArrayRef<BlockArgument> getOutputBlockArguments() {
      return getBody()->getArguments().drop_front(getRank());
    }

    ::mlir::ValueRange getInductionVars() {
      return getBody()->getArguments().take_front(getRank());
    }

    ::mlir::Value getInductionVar(int64_t idx) {
      return getInductionVars()[idx];
    }

    ::mlir::Block::BlockArgListType getRegionOutArgs() {
      return getBody()->getArguments().drop_front(getRank());
    }

    /// Checks if the lbs are zeros and steps are ones.
    bool isNormalized();

    // The ensureTerminator method generated by SingleBlockImplicitTerminator is
    // unaware of the fact that our terminator also needs a region to be
    // well-formed. We override it here to ensure that we do the right thing.
    static void ensureTerminator(Region & region, OpBuilder & builder,
                                 Location loc);

    InParallelOp getTerminator();

    // Declare the shared_outs as inits/outs to DestinationStyleOpInterface.
    MutableOperandRange getDpsInitsMutable() { return getOutputsMutable(); }
  }];
}

//===----------------------------------------------------------------------===//
// InParallelOp
//===----------------------------------------------------------------------===//

def InParallelOp : SCF_Op<"forall.in_parallel", [
       Pure,
       Terminator,
       DeclareOpInterfaceMethods<ParallelCombiningOpInterface>,
       HasParent<"ForallOp">,
      ] # GraphRegionNoTerminator.traits> {
  let summary = "terminates a `forall` block";
  let description = [{
    `scf.forall.in_parallel` is a designated terminator for
    the `scf.forall` operation.

    It has a single region with a single block that contains a flat list of ops.
    Each such op participates in the aggregate formation of a single result of
    the enclosing `scf.forall`.
    The result number corresponds to the position of the op in the terminator.
  }];

  let regions = (region SizedRegion<1>:$region);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  // The default builder does not add a region with an empty body, add our own.
  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins)>,
  ];

  // TODO: Add a `InParallelOpInterface` interface for ops that can
  // appear inside in_parallel.
  let extraClassDeclaration = [{
    ::llvm::SmallVector<::mlir::BlockArgument> getDests();
    ::llvm::iterator_range<::mlir::Block::iterator> getYieldingOps();
    ::mlir::OpResult getParentResult(int64_t idx);
  }];
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

def IfOp : SCF_Op<"if", [DeclareOpInterfaceMethods<RegionBranchOpInterface, [
    "getNumRegionInvocations", "getRegionInvocationBounds",
    "getEntrySuccessorRegions"]>,
    InferTypeOpAdaptor, SingleBlockImplicitTerminator<"scf::YieldOp">,
    RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "if-then-else operation";
  let description = [{
    The `scf.if` operation represents an if-then-else construct for
    conditionally executing two regions of code. The operand to an if operation
    is a boolean value. For example:

    ```mlir
    scf.if %b  {
      ...
    } else {
      ...
    }
    ```

    `scf.if` may also produce results. Which values are returned depends on
    which execution path is taken.

    Example:

    ```mlir
    %x, %y = scf.if %b -> (f32, f32) {
      %x_true = ...
      %y_true = ...
      scf.yield %x_true, %y_true : f32, f32
    } else {
      %x_false = ...
      %y_false = ...
      scf.yield %x_false, %y_false : f32, f32
    }
    ```

    The "then" region has exactly 1 block. The "else" region may have 0 or 1
    block. In case the `scf.if` produces results, the "else" region must also
    have exactly 1 block.

    The blocks are always terminated with `scf.yield`. If `scf.if` defines no
    values, the `scf.yield` can be left out, and will be inserted implicitly.
    Otherwise, it must be explicit.

    Example:

    ```mlir
    scf.if %b  {
      ...
    }
    ```

    The types of the yielded values must match the result types of the
    `scf.if`.
  }];
  let arguments = (ins I1:$condition);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        MaxSizedRegion<1>:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$addThenBlock, "bool":$addElseBlock)>,
    OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$cond,
      "bool":$withElseRegion)>,
    OpBuilder<(ins "Value":$cond,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "buildTerminatedBody">:$thenBuilder,
      CArg<"function_ref<void(OpBuilder &, Location)>",
           "nullptr">:$elseBuilder)>,
  ];

  let extraClassDeclaration = [{
    OpBuilder getThenBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(0);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                                  : OpBuilder::atBlockEnd(body, listener);
    }
    OpBuilder getElseBodyBuilder(OpBuilder::Listener *listener = nullptr) {
      Block* body = getBody(1);
      return getResults().empty() ? OpBuilder::atBlockTerminator(body, listener)
                                  : OpBuilder::atBlockEnd(body, listener);
    }
    Block* thenBlock();
    YieldOp thenYield();
    Block* elseBlock();
    YieldOp elseYield();
  }];
  let hasFolder = 1;
  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ParallelOp
//===----------------------------------------------------------------------===//

def ParallelOp : SCF_Op<"parallel",
    [AutomaticAllocationScope,
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface, ["getSingleInductionVar",
          "getSingleLowerBound", "getSingleUpperBound", "getSingleStep"]>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"scf::YieldOp">]> {
  let summary = "parallel for operation";
  let description = [{
    The "scf.parallel" operation represents a loop nest taking 4 groups of SSA
    values as operands that represent the lower bounds, upper bounds, steps and
    initial values, respectively. The operation defines a variadic number of
    SSA values for its induction variables. It has one region capturing the
    loop body. The induction variables are represented as an argument of this
    region. These SSA values always have type index, which is the size of the
    machine word. The steps are values of type index, required to be positive.
    The lower and upper bounds specify a half-open range: the range includes
    the lower bound but does not include the upper bound. The initial values
    have the same types as results of "scf.parallel". If there are no results,
    the keyword `init` can be omitted.

    Semantically we require that the iteration space can be iterated in any
    order, and the loop body can be executed in parallel. If there are data
    races, the behavior is undefined.

    The parallel loop operation supports reduction of values produced by
    individual iterations into a single result. This is modeled using the
    scf.reduce operation (see scf.reduce for details). Each result of a
    scf.parallel operation is associated with an initial value operand and
    reduce operation that is an immediate child. Reductions are matched to
    result and initial values in order of their appearance in the body.
    Consequently, we require that the body region has the same number of
    results and initial values as it has reduce operations.

    The body region must contain exactly one block that terminates with
    "scf.yield" without operands. Parsing ParallelOp will create such a region
    and insert the terminator when it is absent from the custom format.

    Example:

    ```mlir
    %init = arith.constant 0.0 : f32
    scf.parallel (%iv) = (%lb) to (%ub) step (%step) init (%init) -> f32 {
      %elem_to_reduce = load %buffer[%iv] : memref<100xf32>
      scf.reduce(%elem_to_reduce) : f32 {
        ^bb0(%lhs : f32, %rhs: f32):
          %res = arith.addf %lhs, %rhs : f32
          scf.reduce.return %res : f32
      }
    }
    ```
  }];

  let arguments = (ins Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<Index>:$step,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps, "ValueRange":$initVals,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$steps,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    ValueRange getInductionVars() {
      return getBody()->getArguments();
    }
    unsigned getNumLoops() { return getStep().size(); }
    unsigned getNumReductions() { return getInitVals().size(); }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

def ReduceOp : SCF_Op<"reduce", [HasParent<"ParallelOp">]> {
  let summary = "reduce operation for parallel for";
  let description = [{
    "scf.reduce" is an operation occurring inside "scf.parallel" operations.
    It consists of one block with two arguments which have the same type as the
    operand of "scf.reduce".

    "scf.reduce" is used to model the value for reduction computations of a
    "scf.parallel" operation. It has to appear as an immediate child of a
    "scf.parallel" and is associated with a result value of its parent
    operation.

    Association is in the order of appearance in the body where the first
    result of a parallel loop operation corresponds to the first "scf.reduce"
    in the operation's body region. The reduce operation takes a single
    operand, which is the value to be used in the reduction.

    The reduce operation contains a region whose entry block expects two
    arguments of the same type as the operand. As the iteration order of the
    parallel loop and hence reduction order is unspecified, the result of
    reduction may be non-deterministic unless the operation is associative and
    commutative.

    The result of the reduce operation's body must have the same type as the
    operands and associated result value of the parallel loop operation.
    Example:

    ```mlir
    %operand = arith.constant 1.0 : f32
    scf.reduce(%operand) : f32 {
      ^bb0(%lhs : f32, %rhs: f32):
        %res = arith.addf %lhs, %rhs : f32
        scf.reduce.return %res : f32
    }
    ```
  }];

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$operand,
      CArg<"function_ref<void (OpBuilder &, Location, Value, Value)>",
           "nullptr">:$bodyBuilderFn)>
  ];

  let arguments = (ins AnyType:$operand);
  let hasCustomAssemblyFormat = 1;
  let regions = (region SizedRegion<1>:$reductionOperator);
  let hasRegionVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReduceReturnOp
//===----------------------------------------------------------------------===//

def ReduceReturnOp :
    SCF_Op<"reduce.return", [HasParent<"ReduceOp">, Pure,
                              Terminator]> {
  let summary = "terminator for reduce operation";
  let description = [{
    "scf.reduce.return" is a special terminator operation for the block inside
    "scf.reduce". It terminates the region. It should have the same type as
    the operand of "scf.reduce". Example for the custom format:

    ```mlir
    scf.reduce.return %res : f32
    ```
  }];

  let arguments = (ins AnyType:$result);
  let assemblyFormat = "$result attr-dict `:` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// WhileOp
//===----------------------------------------------------------------------===//

def WhileOp : SCF_Op<"while",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface,
        ["getEntrySuccessorOperands"]>,
     DeclareOpInterfaceMethods<LoopLikeOpInterface, ["getRegionIterArgs"]>,
     RecursiveMemoryEffects, SingleBlock]> {
  let summary = "a generic 'while' loop";
  let description = [{
    This operation represents a generic "while"/"do-while" loop that keeps
    iterating as long as a condition is satisfied. There is no restriction on
    the complexity of the condition. It consists of two regions (with single
    block each): "before" region and "after" region. The names of regions
    indicates whether they execute before or after the condition check.
    Therefore, if the main loop payload is located in the "before" region, the
    operation is a "do-while" loop. Otherwise, it is a "while" loop.

    The "before" region terminates with a special operation, `scf.condition`,
    that accepts as its first operand an `i1` value indicating whether to
    proceed to the "after" region (value is `true`) or not. The two regions
    communicate by means of region arguments. Initially, the "before" region
    accepts as arguments the operands of the `scf.while` operation and uses them
    to evaluate the condition. It forwards the trailing, non-condition operands
    of the `scf.condition` terminator either to the "after" region if the
    control flow is transferred there or to results of the `scf.while` operation
    otherwise. The "after" region takes as arguments the values produced by the
    "before" region and uses `scf.yield` to supply new arguments for the
    "before" region, into which it transfers the control flow unconditionally.

    A simple "while" loop can be represented as follows.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "while" loop, this region computes the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Forward the argument (as result or "after" region argument).
      scf.condition(%condition) %arg1 : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // In a "while" loop, this region is the loop body.
      %next = call @payload(%arg2) : (f32) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %next : f32
    }
    ```

    A simple "do-while" loop can be represented by reducing the "after" block
    to a simple forwarder.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> f32 {
      // "Before" region.
      // In a "do-while" loop, this region contains the loop body.
      %next = call @payload(%arg1) : (f32) -> f32

      // And also evaluates the condition.
      %condition = call @evaluate_condition(%arg1) : (f32) -> i1

      // Loop through the "after" region.
      scf.condition(%condition) %next : f32

    } do {
    ^bb0(%arg2: f32):
      // "After" region.
      // Forwards the values back to "before" region unmodified.
      scf.yield %arg2 : f32
    }
    ```

    Note that the types of region arguments need not to match with each other.
    The op expects the operand types to match with argument types of the
    "before" region; the result types to match with the trailing operand types
    of the terminator of the "before" region, and with the argument types of the
    "after" region. The following scheme can be used to share the results of
    some operations executed in the "before" region with the "after" region,
    avoiding the need to recompute them.

    ```mlir
    %res = scf.while (%arg1 = %init1) : (f32) -> i64 {
      // One can perform some computations, e.g., necessary to evaluate the
      // condition, in the "before" region and forward their results to the
      // "after" region.
      %shared = call @shared_compute(%arg1) : (f32) -> i64

      // Evaluate the condition.
      %condition = call @evaluate_condition(%arg1, %shared) : (f32, i64) -> i1

      // Forward the result of the shared computation to the "after" region.
      // The types must match the arguments of the "after" region as well as
      // those of the `scf.while` results.
      scf.condition(%condition) %shared : i64

    } do {
    ^bb0(%arg2: i64) {
      // Use the partial result to compute the rest of the payload in the
      // "after" region.
      %res = call @payload(%arg2) : (i64) -> f32

      // Forward the new value to the "before" region.
      // The operand types must match the types of the `scf.while` operands.
      scf.yield %res : f32
    }
    ```

    The custom syntax for this operation is as follows.

    ```
    op ::= `scf.while` assignments `:` function-type region `do` region
           `attributes` attribute-dict
    initializer ::= /* empty */ | `(` assignment-list `)`
    assignment-list ::= assignment | assignment `,` assignment-list
    assignment ::= ssa-value `=` ssa-value
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$inits);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$before, SizedRegion<1>:$after);

  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "ValueRange":$operands,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$beforeBuilder,
      "function_ref<void(OpBuilder &, Location, ValueRange)>":$afterBuilder)>
  ];

  let extraClassDeclaration = [{
    using BodyBuilderFn =
        function_ref<void(OpBuilder &, Location, ValueRange)>;

    ConditionOp getConditionOp();
    YieldOp getYieldOp();

    /// Return the values that are yielded from the "after" region (by the
    /// scf.yield op).
    ValueRange getYieldedValues();

    Block::BlockArgListType getBeforeArguments();
    Block::BlockArgListType getAfterArguments();
    Block *getBeforeBody() { return &getBefore().front(); }
    Block *getAfterBody() { return &getAfter().front(); }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IndexSwitchOp
//===----------------------------------------------------------------------===//

def IndexSwitchOp : SCF_Op<"index_switch", [RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"scf::YieldOp">,
    DeclareOpInterfaceMethods<RegionBranchOpInterface,
                              ["getRegionInvocationBounds",
                               "getEntrySuccessorRegions"]>]> {
  let summary = "switch-case operation on an index argument";
  let description = [{
    The `scf.index_switch` is a control-flow operation that branches to one of
    the given regions based on the values of the argument and the cases. The
    argument is always of type `index`.

    The operation always has a "default" region and any number of case regions
    denoted by integer constants. Control-flow transfers to the case region
    whose constant value equals the value of the argument. If the argument does
    not equal any of the case values, control-flow transfer to the "default"
    region.

    Example:

    ```mlir
    %0 = scf.index_switch %arg0 : index -> i32
    case 2 {
      %1 = arith.constant 10 : i32
      scf.yield %1 : i32
    }
    case 5 {
      %2 = arith.constant 20 : i32
      scf.yield %2 : i32
    }
    default {
      %3 = arith.constant 30 : i32
      scf.yield %3 : i32
    }
    ```
  }];

  let arguments = (ins Index:$arg, DenseI64ArrayAttr:$cases);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$defaultRegion,
                        VariadicRegion<SizedRegion<1>>:$caseRegions);

  let assemblyFormat = [{
    $arg attr-dict (`->` type($results)^)?
    custom<SwitchCases>($cases, $caseRegions) `\n`
    `` `default` $defaultRegion
  }];

  let extraClassDeclaration = [{
    /// Get the number of cases.
    unsigned getNumCases();

    /// Get the default region body.
    Block &getDefaultBlock();

    /// Get the body of a case region.
    Block &getCaseBlock(unsigned idx);
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

def YieldOp : SCF_Op<"yield", [Pure, ReturnLike, Terminator,
    ParentOneOf<["ExecuteRegionOp, ForOp", "IfOp", "IndexSwitchOp",
                 "ParallelOp", "WhileOp"]>]> {
  let summary = "loop yield and termination operation";
  let description = [{
    "scf.yield" yields an SSA value from the SCF dialect op region and
    terminates the regions. The semantics of how the values are yielded is
    defined by the parent operation.
    If "scf.yield" has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the "scf.yield" may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

#endif // MLIR_DIALECT_SCF_SCFOPS
