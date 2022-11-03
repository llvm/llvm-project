//===-- GPUOps.td - GPU dialect operation definitions ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the GPU dialect.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_OPS
#define GPU_OPS

include "mlir/Dialect/DLTI/DLTIBase.td"
include "mlir/Dialect/GPU/IR/GPUBase.td"
include "mlir/Dialect/GPU/IR/ParallelLoopMapperAttr.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/Interfaces/InferIntRangeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// GPU Dialect operations.
//===----------------------------------------------------------------------===//

class GPU_Op<string mnemonic, list<Trait> traits = []> :
    Op<GPU_Dialect, mnemonic, traits>;

def GPU_Dimension : I32EnumAttr<"Dimension",
    "a dimension, either 'x', 'y', or 'z'",
    [
      I32EnumAttrCase<"x", 0>,
      I32EnumAttrCase<"y", 1>,
      I32EnumAttrCase<"z", 2>
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def GPU_DimensionAttr : EnumAttr<GPU_Dialect, GPU_Dimension, "dim">;

class GPU_IndexOp<string mnemonic, list<Trait> traits = []> :
    GPU_Op<mnemonic, !listconcat(traits, [
        Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>])>,
    Arguments<(ins GPU_DimensionAttr:$dimension)>, Results<(outs Index)> {
  let assemblyFormat = "$dimension attr-dict";
}

def GPU_BlockDimOp : GPU_IndexOp<"block_dim"> {
  let description = [{
    Returns the number of threads in the thread block (aka the block size) along
    the x, y, or z `dimension`.

    Example:

    ```mlir
    %bDimX = gpu.block_dim x
    ```
  }];
}
def GPU_BlockIdOp : GPU_IndexOp<"block_id"> {
  let description = [{
    Returns the block id, i.e. the index of the current block within the grid
    along the x, y, or z `dimension`.

    Example:

    ```mlir
    %bIdY = gpu.block_id y
    ```
  }];
}
def GPU_GridDimOp : GPU_IndexOp<"grid_dim"> {
  let description = [{
    Returns the number of thread blocks in the grid along the x, y, or z
    `dimension`.

    Example:

    ```mlir
    %gDimZ = gpu.grid_dim z
    ```
  }];
}
def GPU_ThreadIdOp : GPU_IndexOp<"thread_id"> {
  let description = [{
    Returns the thread id, i.e. the index of the current thread within the block
    along the x, y, or z `dimension`.

    Example:

    ```mlir
    %tIdX = gpu.thread_id x
    ```
  }];
}

def GPU_LaneIdOp : GPU_Op<"lane_id", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>]> {
  let description = [{
    Returns the lane id within the subgroup (warp/wave).

    Example:
    ```mlir
    %laneId = gpu.lane_id
    ```
  }];
  let results = (outs Index:$result);
  let assemblyFormat = "attr-dict";
}

def GPU_SubgroupIdOp : GPU_Op<"subgroup_id", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins)>, Results<(outs Index:$result)> {
  let description = [{
    Returns the subgroup id, i.e. the index of the current subgroup within the
    workgroup.

    Example:

    ```mlir
    %sgId = gpu.subgroup_id : index
    ```
  }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def GPU_GlobalIdOp : GPU_IndexOp<"global_id"> {
  let description = [{
    Returns the unique global workitem/thread id, i.e., the unique index of the
    current workitem/thread within all workgroups / grid along the x, y, or z
    `dimension`.

    Example:

    ```mlir
    %gidX = gpu.global_id x
    ```
  }];
}


def GPU_NumSubgroupsOp : GPU_Op<"num_subgroups", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins)>, Results<(outs Index:$result)> {
  let description = [{
    Returns the number of subgroups within a workgroup.

    Example:

    ```mlir
    %numSg = gpu.num_subgroups : index
    ```
  }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def GPU_SubgroupSizeOp : GPU_Op<"subgroup_size", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins)>, Results<(outs Index:$result)> {
  let description = [{
    Returns the number of threads within a subgroup.

    Example:

    ```mlir
    %sgSz = gpu.subgroup_size : index
    ```
  }];

  let assemblyFormat = "attr-dict `:` type($result)";
}

def GPU_GPUFuncOp : GPU_Op<"func", [
    HasParent<"GPUModuleOp">, AutomaticAllocationScope, FunctionOpInterface,
    IsolatedFromAbove, Symbol
  ]> {
  let summary = "Function executable on a GPU";

  let description = [{
    Defines a function that can be executed on a GPU. This supports memory
    attribution and its body has a particular execution model.

    GPU functions are either kernels (as indicated by the `kernel` attribute) or
    regular functions. The former can be launched from the host side, while the
    latter are device side only.

    The memory attribution defines SSA values that correspond to memory buffers
    allocated in the memory hierarchy of the GPU (see below).

    The operation has one attached region that corresponds to the body of the
    function. The region arguments consist of the function arguments without
    modification, followed by buffers defined in memory annotations. The body of
    a GPU function, when launched, is executed by multiple work items. There are
    no guarantees on the order in which work items execute, or on the connection
    between them. In particular, work items are not necessarily executed in
    lock-step. Synchronization ops such as "gpu.barrier" should be used to
    coordinate work items. Declarations of GPU functions, i.e. not having the
    body region, are not supported.

    Syntax:

    ```
    op ::= `gpu.func` symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)?
           memory-attribution `kernel`? function-attributes? region

    memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                           (`private` `(` ssa-id-and-type-list `)`)?
    ```

    Example:

    ```mlir
    gpu.func @foo(%arg0: index)
        workgroup(%workgroup: memref<32xf32, 3>)
        private(%private: memref<1xf32, 5>)
        kernel
        attributes {qux: "quux"} {
      gpu.return
    }
    ```

    The generic form illustrates the concept

    ```mlir
    "gpu.func"(%arg: index) {sym_name: "foo", kernel, qux: "quux"} ({
    ^bb0(%arg0: index, %workgroup: memref<32xf32, 3>,
         %private: memref<1xf32, 5>):
      "gpu.return"() : () -> ()
    }) : (index) -> ()
    ```

    Note the non-default memory spaces used in memref types in memory
    attribution.
  }];

  let arguments = (ins TypeAttrOf<FunctionType>:$function_type);
  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "StringRef":$name, "FunctionType":$type,
      CArg<"TypeRange", "{}">:$workgroupAttributions,
      CArg<"TypeRange", "{}">:$privateAttributions,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    /// Returns `true` if the GPU function defined by this Op is a kernel, i.e.
    /// it is intended to be launched from host.
    bool isKernel() {
      return (*this)->getAttrOfType<UnitAttr>(
          GPUDialect::getKernelFuncAttrName()) != nullptr;
    }

    /// Returns the number of buffers located in the workgroup memory.
    unsigned getNumWorkgroupAttributions() {
      auto attr = (*this)->getAttrOfType<IntegerAttr>(
          getNumWorkgroupAttributionsAttrName());
      return attr ? attr.getInt() : 0;
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the workgroup memory
    ArrayRef<BlockArgument> getWorkgroupAttributions() {
      auto begin =
          std::next(getBody().args_begin(), getFunctionType().getNumInputs());
      auto end = std::next(begin, getNumWorkgroupAttributions());
      return {begin, end};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// workgroup memory.
    BlockArgument addWorkgroupAttribution(Type type, Location loc);

    /// Returns the number of buffers located in the private memory.
    unsigned getNumPrivateAttributions() {
      return getBody().getNumArguments() - getFunctionType().getNumInputs() -
          getNumWorkgroupAttributions();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the private memory.
    ArrayRef<BlockArgument> getPrivateAttributions() {
      // Buffers on the private memory always come after buffers on the workgroup
      // memory.
      auto begin =
          std::next(getBody().args_begin(),
                    getFunctionType().getNumInputs() + getNumWorkgroupAttributions());
      return {begin, getBody().args_end()};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// private memory.
    BlockArgument addPrivateAttribution(Type type, Location loc);

    /// Returns the name of the attribute containing the number of buffers
    /// located in the workgroup memory.
    static StringRef getNumWorkgroupAttributionsAttrName() {
      return "workgroup_attributions";
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the keywords used in the custom syntax for this Op.
    static StringRef getWorkgroupKeyword() { return "workgroup"; }
    static StringRef getPrivateKeyword() { return "private"; }
    static StringRef getKernelKeyword() { return "kernel"; }

    /// Hook for FunctionOpInterface verifier.
    LogicalResult verifyType();

    /// Verifies the body of the function.
    LogicalResult verifyBody();
  }];
  let hasCustomAssemblyFormat = 1;
}

def GPU_LaunchFuncOp : GPU_Op<"launch_func",
                              [GPU_AsyncOpInterface, AttrSizedOperandSegments]>,
    Arguments<(ins Variadic<GPU_AsyncToken>:$asyncDependencies,
               SymbolRefAttr:$kernel,
               Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
               Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
               Optional<I32>:$dynamicSharedMemorySize,
               Variadic<AnyType>:$kernelOperands)>,
    Results<(outs Optional<GPU_AsyncToken>:$asyncToken)> {
  let summary = "Launches a function as a GPU kernel";

  let description = [{
    Launch a kernel function on the specified grid of thread blocks.
    `gpu.launch` operations are lowered to `gpu.launch_func` operations by
    outlining the kernel body into a function in a dedicated module, which
    reflects the separate compilation process. The kernel function is required
    to have the `gpu.kernel` attribute. The module containing the kernel
    function is required to be a gpu.module. And finally, the module containing
    the kernel module (which thus cannot be the top-level module) is required
    to have the `gpu.container_module` attribute. The `gpu.launch_func`
    operation has a symbol attribute named `kernel` to identify the fully
    specified kernel function to launch (both the gpu.module and func).

    The `gpu.launch_func` supports async dependencies: the kernel does not start
    executing until the ops producing those async dependencies have completed.

    By the default, the host implicitly blocks until kernel execution has
    completed. If the `async` keyword is present, the host does not block but
    instead a `!gpu.async.token` is returned. Other async GPU ops can take this
    token as dependency.

    The operation requires at least the grid and block sizes along the x,y,z
    dimensions as arguments. When a lower-dimensional kernel is required,
    unused sizes must be explicitly set to `1`.

    The remaining operands are optional. The first optional operand corresponds
    to the amount of dynamic shared memory a kernel's workgroup should be
    allocated; when this operand is not present, a zero size is assumed.

    The remaining operands if present are passed as arguments to the kernel
    function.

    Example:

    ```mlir
    module attributes {gpu.container_module} {

      // This module creates a separate compilation unit for the GPU compiler.
      gpu.module @kernels {
        func.func @kernel_1(%arg0 : f32, %arg1 : memref<?xf32, 1>)
            attributes { nvvm.kernel = true } {

          // Operations that produce block/thread IDs and dimensions are
          // injected when outlining the `gpu.launch` body to a function called
          // by `gpu.launch_func`.
          %tIdX = gpu.thread_id x
          %tIdY = gpu.thread_id y
          %tIdZ = gpu.thread_id z

          %bDimX = gpu.block_dim x
          %bDimY = gpu.block_dim y
          %bDimZ = gpu.block_dim z

          %bIdX = gpu.block_id x
          %bIdY = gpu.block_id y
          %bIdZ = gpu.block_id z

          %gDimX = gpu.grid_dim x
          %gDimY = gpu.grid_dim y
          %gDimZ = gpu.grid_dim z

          "some_op"(%bx, %tx) : (index, index) -> ()
          %42 = load %arg1[%bx] : memref<?xf32, 1>
        }
      }

      %t0 = gpu.wait async
      gpu.launch_func
          async                           // (Optional) Don't block host, return token.
          [%t0]                           // (Optional) Execute only after %t0 has completed.
          @kernels::@kernel_1             // Kernel function.
          blocks in (%cst, %cst, %cst)    // Grid size.
          threads in (%cst, %cst, %cst)   // Block size.
          dynamic_shared_memory_size %s   // (Optional) Amount of dynamic shared
                                          // memory to allocate for a workgroup.
          args(%arg0 : f32,               // (Optional) Kernel arguments.
               %arg1 : memref<?xf32, 1>)
    }
    ```
  }];

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "GPUFuncOp":$kernelFunc, "KernelDim3":$gridSize,
      "KernelDim3":$blockSize, "Value":$dynamicSharedMemorySize,
      "ValueRange":$kernelOperands,
      CArg<"Type", "nullptr">:$asyncTokenType,
      CArg<"ValueRange", "{}">:$asyncDependencies)>
  ];

  let extraClassDeclaration = [{
    /// The name of the kernel's containing module.
    StringAttr getKernelModuleName();

    /// The name of the kernel.
    StringAttr getKernelName();

    /// The number of operands passed to the kernel function.
    unsigned getNumKernelOperands();

    /// The i-th operand passed to the kernel function.
    Value getKernelOperand(unsigned i);

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();

    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();

    // This needs to quietly verify if attributes with names defined below are
    // present since it is run before the verifier of this op.
    friend LogicalResult GPUDialect::verifyOperationAttribute(Operation *,
                                                              NamedAttribute);
  }];

  let assemblyFormat = [{
      custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
      $kernel
      `blocks` `in` ` ` `(`$gridSizeX`,` $gridSizeY`,` $gridSizeZ`)`
      `threads` `in` ` ` `(`$blockSizeX`,` $blockSizeY`,` $blockSizeZ`)`
      (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
      custom<LaunchFuncOperands>($kernelOperands, type($kernelOperands)) attr-dict
  }];
  let hasVerifier = 1;
}

def GPU_LaunchOp : GPU_Op<"launch", [
      AutomaticAllocationScope, AttrSizedOperandSegments, GPU_AsyncOpInterface,
      DeclareOpInterfaceMethods<InferIntRangeInterface>]>,
    Arguments<(ins Variadic<GPU_AsyncToken>:$asyncDependencies,
               Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
               Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
               Optional<I32>:$dynamicSharedMemorySize)>,
    Results<(outs Optional<GPU_AsyncToken>:$asyncToken)> {
  let summary = "GPU kernel launch operation";

  let description = [{
    Launch a kernel on the specified grid of thread blocks. The body of the
    kernel is defined by the single region that this operation contains. The
    operation takes an optional list of async dependencies followed by six
    operands and an optional operand.

    The `async` keyword indicates the kernel should be launched asynchronously;
    the operation returns a new !gpu.async.token when the keyword is specified.
    The kernel launched does not start executing until the ops producing its
    async dependencies (optional operands) have completed.

    The first three operands (following any async dependencies) are grid sizes
    along the x,y,z dimensions and the following three are block sizes along the
    x,y,z dimensions. When a lower-dimensional kernel is required, unused sizes
    must be explicitly set to `1`.  The last operand is optional and corresponds
    to the amount of dynamic shared memory a kernel's workgroup should be
    allocated; when this operand is not present, a zero size is assumed.

    The body region has _twelve_ arguments, grouped as follows:

    -   three arguments that contain block identifiers along x,y,z dimensions;
    -   three arguments that contain thread identifiers along x,y,z dimensions;
    -   operands of the `gpu.launch` operation as is (i.e. the operands for
        grid and block sizes).

    Syntax:

    ```
    operation ::= `gpu.launch` (`async` (`[` ssa-id-list `]`)? )?
                             `block` `(` ssa-id-list `)` `in` ssa-reassignment
                             `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                             (dynamic_shared_memory_size ssa-use)?
                             region attr-dict?
    ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
    ```

    Example:

    ```mlir
    gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
               threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5) {
      // Block and thread identifiers, as well as block/grid sizes are
      // immediately usable inside body region.
      "some_op"(%bx, %tx) : (index, index) -> ()
      // Assuming %val1 is defined outside the gpu.launch region.
      %42 = load %val1[%bx] : memref<?xf32, 1>
    }

    // Generic syntax explains how the pretty syntax maps to the IR structure.
    "gpu.launch"(%cst, %cst, %c1,  // Grid sizes.
                 %cst, %c1, %c1)   // Block sizes.

        {/*attributes*/}
        // All sizes and identifiers have "index" size.
        : (index, index, index, index, index, index) -> () {
    // The operation passes block and thread identifiers, followed by grid and
    // block sizes.
    ^bb0(%bx : index, %by : index, %bz : index,
         %tx : index, %ty : index, %tz : index,
         %num_bx : index, %num_by : index, %num_bz : index,
         %num_tx : index, %num_ty : index, %num_tz : index)
      "some_op"(%bx, %tx) : (index, index) -> ()
      %3 = "memref.load"(%val1, %bx) : (memref<?xf32, 1>, index) -> f32
    }
    ```

    Rationale: using operation/block arguments gives analyses a clear way of
    understanding that a value has additional semantics (e.g., we will need to
    know what value corresponds to threadIdx.x for coalescing). We can recover
    these properties by analyzing the operations producing values, but it is
    easier just to have that information by construction.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins "Value":$gridSizeX, "Value":$gridSizeY,
      "Value":$gridSizeZ, "Value":$blockSizeX, "Value":$blockSizeY,
      "Value":$blockSizeZ,
      CArg<"Value", "nullptr">:$dynamicSharedMemorySize,
      CArg<"Type", "nullptr">:$asyncTokenType,
      CArg<"ValueRange", "{}">:$asyncDependencies)>
  ];

  let extraClassDeclaration = [{
    /// Get the SSA values corresponding to kernel block identifiers.
    KernelDim3 getBlockIds();
    /// Get the SSA values corresponding to kernel thread identifiers.
    KernelDim3 getThreadIds();
    /// Get the SSA values corresponding to kernel grid size.
    KernelDim3 getGridSize();
    /// Get the SSA values corresponding to kernel block size.
    KernelDim3 getBlockSize();

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();
    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();

    static StringRef getBlocksKeyword() { return "blocks"; }
    static StringRef getThreadsKeyword() { return "threads"; }
    static StringRef getDynamicSharedMemorySizeKeyword() {
      return "dynamic_shared_memory_size";
    }

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    static constexpr unsigned kNumConfigOperands = 6;

    /// The number of region attributes containing the launch configuration,
    /// placed in the leading positions of the argument list.
    static constexpr unsigned kNumConfigRegionAttributes = 12;
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
}

def GPU_PrintfOp : GPU_Op<"printf", [MemoryEffects<[MemWrite]>]>,
  Arguments<(ins StrAttr:$format,
                Variadic<AnyTypeOf<[AnyInteger, Index, AnyFloat]>>:$args)> {
  let summary = "Device-side printf, as in CUDA or OpenCL, for debugging";
  let description = [{
    `gpu.printf` takes a literal format string `format` and an arbitrary number of
    scalar arguments that should be printed.

    The format string is a C-style printf string, subject to any restrictions
    imposed by one's target platform.
  }];
  let assemblyFormat = [{
    $format attr-dict ($args^ `:` type($args))?
  }];
}

def GPU_ReturnOp : GPU_Op<"return", [HasParent<"GPUFuncOp">, Pure,
                                     Terminator]>,
    Arguments<(ins Variadic<AnyType>:$operands)>, Results<(outs)> {
  let summary = "Terminator for GPU functions.";
  let description = [{
    A terminator operation for regions that appear in the body of  `gpu.func`
    functions. The operands to the `gpu.return` are the result values returned
    by an invocation of the `gpu.func`.
  }];

  let builders = [OpBuilder<(ins), [{ // empty}]>];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def GPU_TerminatorOp : GPU_Op<"terminator", [HasParent<"LaunchOp">,
                                             Pure, Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for GPU launch regions.";
  let description = [{
    A terminator operation for regions that appear in the body of `gpu.launch`
    operation.  These regions are not expected to return any value so the
    terminator takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}

def GPU_YieldOp : GPU_Op<"yield", [Pure, Terminator]>,
    Arguments<(ins Variadic<AnyType>:$values)> {
  let summary = "GPU yield operation";
  let description = [{
    gpu.yield` is a special terminator operation for blocks inside regions
    in gpu ops. It returns values to the immediately enclosing gpu op.

    Example:

    ```mlir
    gpu.yield %f0, %f1 : f32, f32
    ```
  }];
}

// add, mul mirror the XLA ComparisonDirection enum.
def GPU_AllReduceOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
def GPU_AllReduceOpAnd : I32EnumAttrCase<"AND", 1, "and">;
def GPU_AllReduceOpMax : I32EnumAttrCase<"MAX", 2, "max">;
def GPU_AllReduceOpMin : I32EnumAttrCase<"MIN", 3, "min">;
def GPU_AllReduceOpMul : I32EnumAttrCase<"MUL", 4, "mul">;
def GPU_AllReduceOpOr  : I32EnumAttrCase<"OR",  5, "or">;
def GPU_AllReduceOpXor : I32EnumAttrCase<"XOR", 6, "xor">;

def GPU_AllReduceOperation : I32EnumAttr<"AllReduceOperation",
    "built-in reduction operations supported by gpu.allreduce.",
    [
      GPU_AllReduceOpAdd,
      GPU_AllReduceOpAnd,
      GPU_AllReduceOpMax,
      GPU_AllReduceOpMin,
      GPU_AllReduceOpMul,
      GPU_AllReduceOpOr,
      GPU_AllReduceOpXor
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def GPU_AllReduceOperationAttr : EnumAttr<GPU_Dialect, GPU_AllReduceOperation,
                                          "all_reduce_op">;

def GPU_AllReduceOp : GPU_Op<"all_reduce",
    [SameOperandsAndResultType, IsolatedFromAbove]>,
    Arguments<(ins AnyType:$value,
               OptionalAttr<GPU_AllReduceOperationAttr>:$op)>,
    Results<(outs AnyType)> {
  let summary = "Reduce values among workgroup.";
  let description = [{
    The `all_reduce` op reduces the value of every work item across a local
    workgroup. The result is equal for all work items of a workgroup.

    For example, both

    ```mlir
    %1 = gpu.all_reduce add %0 {} : (f32) -> (f32)
    %2 = gpu.all_reduce %0 {
    ^bb(%lhs : f32, %rhs : f32):
      %sum = arith.addf %lhs, %rhs : f32
      "gpu.yield"(%sum) : (f32) -> ()
    } : (f32) -> (f32)
    ```

    compute the sum of each work item's %0 value. The first version specifies
    the accumulation as operation, whereas the second version specifies the
    accumulation as code region. The accumulation operation must be one of:
    `add`, `and`, `max`, `min`, `mul`, `or`, `xor`.

    Either none or all work items of a workgroup need to execute this op
    in convergence.
  }];
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ custom<AllReduceOperation>($op) $value $body attr-dict
                          `:` functional-type(operands, results) }];
  let hasRegionVerifier = 1;
}

def GPU_SubgroupReduceOp : GPU_Op<"subgroup_reduce",
    [SameOperandsAndResultType]>,
    Arguments<(ins AnyType:$value,
               GPU_AllReduceOperationAttr:$op)>,
    Results<(outs AnyType)> {
  let summary = "Reduce values among subgroup.";
  let description = [{
    The `subgroup_reduce` op reduces the value of every work item across a
    subgroup. The result is equal for all work items of a subgroup.

    Example:

    ```mlir
    %1 = gpu.subgroup_reduce add %0 : (f32) -> (f32)
    ```

    Either none or all work items of a subgroup need to execute this op
    in convergence.
  }];
  let assemblyFormat = [{ custom<AllReduceOperation>($op) $value attr-dict
                          `:` functional-type(operands, results) }];
  let hasVerifier = 1;
}

def GPU_ShuffleOpXor  : I32EnumAttrCase<"XOR",  0, "xor">;
def GPU_ShuffleOpDown : I32EnumAttrCase<"DOWN", 1, "down">;
def GPU_ShuffleOpUp   : I32EnumAttrCase<"UP",   2, "up">;
def GPU_ShuffleOpIdx  : I32EnumAttrCase<"IDX",  3, "idx">;

def GPU_ShuffleMode : I32EnumAttr<"ShuffleMode",
    "Indexing modes supported by gpu.shuffle.",
    [
      GPU_ShuffleOpXor, GPU_ShuffleOpUp, GPU_ShuffleOpDown, GPU_ShuffleOpIdx,
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def GPU_ShuffleModeAttr : EnumAttr<GPU_Dialect, GPU_ShuffleMode,
                                   "shuffle_mode">;

def I32OrF32 : TypeConstraint<Or<[I32.predicate, F32.predicate]>,
                                 "i32 or f32">;

def GPU_ShuffleOp : GPU_Op<
    "shuffle", [Pure, AllTypesMatch<["value", "shuffleResult"]>]>,
    Arguments<(ins I32OrF32:$value, I32:$offset, I32:$width,
               GPU_ShuffleModeAttr:$mode)>,
    Results<(outs I32OrF32:$shuffleResult, I1:$valid)> {
  let summary = "Shuffles values within a subgroup.";
  let description = [{
    The "shuffle" op moves values to a different invocation within the same
    subgroup.

    Example:

    ```mlir
    %1, %2 = gpu.shuffle %0, %offset, %width xor : f32
    ```

    For lane k returns the value from lane `k ^ offset` and `true` if that lane
    is smaller than %width. Otherwise it returns an unspecified value and
    `false`. A lane is the index of an invocation relative to its subgroup.

    The width specifies the number of invocations that participate in the
    shuffle. The width needs to be the same for all invocations that participate
    in the shuffle. Exactly the first `width` invocations of a subgroup need to
    execute this op in convergence.
  }];
  let builders = [
    // Helper function that creates a shuffle with constant offset/width.
    OpBuilder<(ins "Value":$value, "int32_t":$offset, "int32_t":$width,
                   "ShuffleMode":$mode)>
  ];
  let assemblyFormat = "$mode $value `,` $offset `,` $width attr-dict `:` type($value)";
}

def GPU_BarrierOp : GPU_Op<"barrier"> {
  let summary = "Synchronizes all work items of a workgroup.";
  let description = [{
    The "barrier" op synchronizes all work items of a workgroup. It is used
    to coordinate communication between the work items of the workgroup.

    ```mlir
    gpu.barrier
    ```

    waits until all work items in the workgroup have reached this point
    and all memory accesses made by these work items prior to the op are
    visible to all work items in the workgroup. Data hazards between work items
    accessing the same memory can be avoided by synchronizing work items
    in-between these accesses.

    Either none or all work items of a workgroup need to execute this op
    in convergence.
  }];
  let assemblyFormat = "attr-dict";
}

def GPU_GPUModuleOp : GPU_Op<"module", [
  DataLayoutOpInterface, HasDefaultDLTIDataLayout, IsolatedFromAbove,
  SymbolTable, Symbol,
  SingleBlockImplicitTerminator<"ModuleEndOp">
]> {
  let summary = "A top level compilation unit containing code to be run on a GPU.";
  let description = [{
    GPU module contains code that is intended to be run on a GPU. A host device
    can launch this code through a gpu.launc_func that creates a fully
    qualified symbol through the gpu.module's symbol and a gpu.func symbol
    contained in the gpu.module.

    The module's top-level scope is modeled by a single region with a single
    block. GPU modules are required to have a name that is used for symbol
    resolution by the gpu.launch_func operation.

    Using an op with a region to define a GPU module enables "embedding" GPU
    modules with SIMT execution models in other dialects in a clean manner and
    allows filtering of code regions to execute passes on only code intended to
    or not intended to be run on the separate device.

    ```
      gpu.module @symbol_name {
      gpu.func {}
        ...
      gpu.module_end
    }

    ```
  }];
  let builders = [OpBuilder<(ins "StringRef":$name)>];
  let regions = (region SizedRegion<1>:$bodyRegion);
  let hasCustomAssemblyFormat = 1;

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;
}

def GPU_ModuleEndOp : GPU_Op<"module_end", [
  Terminator, HasParent<"GPUModuleOp">
]> {
  let summary = "A pseudo op that marks the end of a gpu.module.";
  let description = [{
    This op terminates the only block inside the only region of a `gpu.module`.
  }];

  let assemblyFormat = "attr-dict";
}

def GPU_HostRegisterOp : GPU_Op<"host_register">,
    Arguments<(ins AnyUnrankedMemRef:$value)> {
  let summary = "Registers a memref for access from device.";
  let description = [{
    This op maps the provided host buffer into the device address space.

    This operation may not be supported in every environment, there is not yet a
    way to check at runtime whether this feature is supported.

    Writes from the host are guaranteed to be visible to device kernels that are
    launched afterwards. Writes from the device are guaranteed to be visible on
    the host after synchronizing with the device kernel completion.
  }];

  let assemblyFormat = "$value attr-dict `:` type($value)";
}

def GPU_WaitOp : GPU_Op<"wait", [GPU_AsyncOpInterface]> {
  let summary = "Wait for async gpu ops to complete.";
  let description = [{
    This op synchronizes the host or the device with a list of dependent ops.

    If the op contains the `async` keyword, it returns a new async token which
    is synchronized with the op arguments. This new token is merely a shortcut
    to the argument list, and one could replace the uses of the result with the
    arguments for the same effect. The async version of this op is primarily
    used to make each async token have a single use during lowering and
    thereby make forks in async execution explicit. Example usage:

    ```mlir
    %t0 = gpu.foo async : !gpu.async.token
    %t1 = gpu.bar async : !gpu.async.token
    %t2 = gpu.wait async [%t0, %t1]
    // gpu.baz doesn't run until gpu.foo and gpu.bar have both completed, just
    // as if the async dependencies were [%t0, %t1].
    %t3 = gpu.baz async [%t2]
    ```

    If the op does not contain the `async` keyword, it does not return a new
    async token but blocks until all ops producing the async dependency tokens
    finished execution. All dependent memory operations are visible to the host
    once this op completes. Example usage:

    ```mlir
    %t0 = gpu.foo async : !gpu.async.token
    %t1 = gpu.bar async : !gpu.async.token
    // The gpu.wait op blocks until gpu.foo and gpu.bar have completed.
    gpu.wait [%t0, %t1]
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) attr-dict
  }];

  let hasCanonicalizer = 1;
}

def GPU_AllocOp : GPU_Op<"alloc", [
    GPU_AsyncOpInterface,
    AttrSizedOperandSegments
  ]> {

  let summary = "GPU memory allocation operation.";
  let description = [{
    The `gpu.alloc` operation allocates a region of memory on the GPU. It is
    similar to the `memref.alloc` op, but supports asynchronous GPU execution.

    The op does not execute before all async dependencies have finished
    executing.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it also returns a !gpu.async.token.

    If the `host_shared` keyword is present, the memory will be allocated in a
    memory accessible both on host and on device.

    Example:

    ```mlir
    %memref, %token = gpu.alloc async [%dep] host_shared (%width) : memref<64x?xf32, 1>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Variadic<Index>:$dynamicSizes, Variadic<Index>:$symbolOperands,
                   UnitAttr:$hostShared);
  let results = (outs Res<AnyMemRef, "", [MemAlloc]>:$memref,
                 Optional<GPU_AsyncToken>:$asyncToken);

  let extraClassDeclaration = [{
    MemRefType getType() { return getMemref().getType().cast<MemRefType>(); }
  }];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) (` ` `host_shared` $hostShared^)? ` `
    `(` $dynamicSizes `)` (`` `[` $symbolOperands^ `]`)? attr-dict `:` type($memref)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def GPU_DeallocOp : GPU_Op<"dealloc", [GPU_AsyncOpInterface]> {

  let summary = "GPU memory deallocation operation";

  let description = [{
    The `gpu.dealloc` operation frees the region of memory referenced by a
    memref which was originally created by the `gpu.alloc` operation. It is
    similar to the `memref.dealloc` op, but supports asynchronous GPU execution.

    The op does not execute before all async dependencies have finished
    executing.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token.

    Example:

    ```mlir
    %token = gpu.dealloc async [%dep] %memref : memref<8x64xf32, 1>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Arg<AnyMemRef, "", [MemFree]>:$memref);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $memref attr-dict `:` type($memref)
  }];
}

def GPU_MemcpyOp : GPU_Op<"memcpy", [GPU_AsyncOpInterface]> {

  let summary = "GPU memcpy operation";

  let description = [{
    The `gpu.memcpy` operation copies the content of one memref to another.

    The op does not execute before all async dependencies have finished
    executing.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token.

    Example:

    ```mlir
    %token = gpu.memcpy async [%dep] %dst, %src : memref<?xf32, 1>, memref<?xf32>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Arg<AnyMemRef, "", [MemWrite]>:$dst,
                   Arg<AnyMemRef, "", [MemRead]>:$src);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $dst`,` $src `:` type($dst)`,` type($src) attr-dict
  }];
  let hasFolder = 1;
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def GPU_MemsetOp : GPU_Op<"memset",
  [GPU_AsyncOpInterface, AllElementTypesMatch<["dst", "value"]>]> {

  let summary = "GPU memset operation";

  let description = [{
    The `gpu.memset` operation sets the content of memref to a scalar value.

    The op does not execute before all async dependencies have finished
    executing.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token.

    Example:

    ```mlir
    %token = gpu.memset async [%dep] %dst, %value : memref<?xf32, 1>, f32
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Arg<AnyMemRef, "", [MemWrite]>:$dst,
                   Arg<AnyType, "">:$value);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $dst`,` $value `:` type($dst)`,` type($value) attr-dict
  }];
  let hasFolder = 1;
}

def GPU_SetDefaultDeviceOp : GPU_Op<"set_default_device",
                                    [MemoryEffects<[MemWrite]>]>,
    Arguments<(ins I32:$devIndex)> {
  let summary = "Set default GPU for operations after this by index";
  let description = [{
    Operation that sets the current default GPU, using a zero-based index
    into the set of GPUs on the system. The default GPU setting may be
    thread-local.
  }];
  let assemblyFormat = "attr-dict $devIndex";
}

def GPU_SubgroupMmaLoadMatrixOp : GPU_Op<"subgroup_mma_load_matrix",
    [MemoryEffects<[MemRead]>]>{

  let summary = "GPU warp synchronous matrix load";

  let description = [{
    The `gpu.subgroup_mma_load_matrix` operation loads a matrix collectively
    using all the threads in a subgroup.

    This operation takes a memref as its first operand: it is the source matrix
    from which data is to be loaded. The op returns a `!gpu.mma_matrix`. The
    source memref can be in global memory or shared memory. The load address is
    determined using `indices`. The matrix being loaded into is the result.  The
    `leadDimension` attribute specifies the leading dimension size of the source
    matrix which eventually allows the lowering to determine the size of each
    row.

    This op is often meant to be used along with `gpu.subgroup_mma_store_matrix` and
    `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
     %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
          : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, "AOp">
    ```
  }];

  let arguments = (ins Arg<GPU_MMAMemRef, "", [MemRead]>:$srcMemref,
                  Variadic<Index>:$indices,
                  IndexAttr:$leadDimension);

  let results = (outs GPU_MMAMatrix:$res);

  let assemblyFormat = [{
    $srcMemref`[`$indices`]` attr-dict `:` type($srcMemref) `->` type($res)
  }];
  let hasVerifier = 1;
}

def GPU_SubgroupMmaStoreMatrixOp : GPU_Op<"subgroup_mma_store_matrix",
    [MemoryEffects<[MemWrite]>]>{

  let summary = "GPU warp synchronous matrix store";

  let description = [{
    The `gpu.subgroup_mma_store_matrix` operation stores a matrix collectively
    using all the threads in a subgroup.

    This operation takes a `!gpu.mma_matrix` and a memref as operands.
    `!gpu.mma_matrix` is the source value containing the data to be stored into the
    destination memref which can be in global or shared memory.  The store address
    is determined using the indices provided. The `leadDimension` attribute
    specifies the leading dimension of the destination matrix.

    This op is often meant to be used along with `gpu.subgroup_mma_load_matrix` and
    `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
    gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                    : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
    ```
  }];

  let arguments = (ins Arg<MMAMatrixOf<[F16, F32]>>:$src,
                  Arg<GPU_MMAMemRef, "",[MemWrite]>:$dstMemref,
                  Variadic<Index>:$indices,
                  IndexAttr:$leadDimension);

  let assemblyFormat = [{
    $src`,` $dstMemref`[`$indices`]` attr-dict `:` type($src)`,` type($dstMemref)
  }];
  let hasVerifier = 1;
}

def GPU_SubgroupMmaComputeOp : GPU_Op<"subgroup_mma_compute",
   [Pure, AllTypesMatch<["opC", "res"]>]>{

  let summary = "GPU warp synchronous matrix multiply accumulate";

  let description = [{
    The `gpu.subgroup_mma_compute` operation performs a matrix-multiply accumulate (mma)
    operation using all the threads in a subgroup.

    This operation takes three `!gpu.mma_matrix`s as arguments: these hold `A`,
     `B` and `C`operands for the mma operation. The operation performed is represented
    as `C += A * B`. The op returns a `!gpu.mma_matrix` which contains the result of
    the operation held by all threads in a subgroup.

    This op is meant to be used along with `gpu.subgroup_mma_store_matrix` and
    `gpu.subgroup_mma_load_matrix` ops.

    Example:

    ```mlir
    %D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
      !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">>
      -> !gpu.mma_matrix<16x16xf16, "COp">
    ```
  }];

  let arguments = (ins Arg<MMAMatrixOf<[F16, F32]>>:$opA,
                  Arg<MMAMatrixOf<[F16, F32]>>:$opB,
                  Arg<MMAMatrixOf<[F16, F32]>>:$opC);

  let results = (outs GPU_MMAMatrix:$res);

  let assemblyFormat = [{
    $opA`,` $opB`,` $opC attr-dict `:` type($opA)`,` type($opB) `->` type($res)
  }];
  let hasVerifier = 1;
}

def GPU_SubgroupMmaConstantMatrixOp : GPU_Op<"subgroup_mma_constant_matrix",
    [Pure,
     TypesMatchWith<"value type matches element type of mma_matrix",
                    "res", "value",
                    "$_self.cast<gpu::MMAMatrixType>().getElementType()">]>{

  let summary = "GPU warp synchronous constant matrix";

  let description = [{
    The `gpu.subgroup_mma_constant_matrix` creates a `!gpu.mma_matrix` with
    constant elements.

    The operation takes a scalar input and return a `!gpu.mma_matrix` where each
    element of is equal to the operand constant. The destination mma_matrix type
    must have elememt type equal to the constant type. Since the layout of
    `!gpu.mma_matrix` is opaque this only support setting all the elements to
    the same value.

    This op is meant to be used along with `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
     %0 = gpu.subgroup_mma_constant_matrix %a :
       !gpu.mma_matrix<16x16xf16, "AOp">
     %1 = gpu.subgroup_mma_constant_matrix %b :
       !gpu.mma_matrix<16x16xf32, "COp">
    ```
  }];

  let arguments = (ins AnyTypeOf<[F16, F32]>:$value);

  let results = (outs GPU_MMAMatrix:$res);

  let extraClassDeclaration = [{
    gpu::MMAMatrixType getType() {
      return getRes().getType().cast<gpu::MMAMatrixType>();
    }
  }];

  let assemblyFormat = [{
    $value attr-dict `:` type($res)
  }];
}

def GPU_ElementwiseOpAddF  : I32EnumAttrCase<"ADDF", 0, "addf">;
def GPU_ElementwiseOpMulF  : I32EnumAttrCase<"MULF", 1, "mulf">;
def GPU_ElementwiseOpSUBF  : I32EnumAttrCase<"SUBF", 2, "subf">;
def GPU_ElementwiseOpMaxF : I32EnumAttrCase<"MAXF", 3, "maxf">;
def GPU_ElementwiseOpMinF : I32EnumAttrCase<"MINF", 4, "minf">;
def GPU_ElementwiseOpDivF : I32EnumAttrCase<"DIVF", 5, "divf">;
def GPU_ElementwiseOpAddI  : I32EnumAttrCase<"ADDI", 6, "addi">;
def GPU_ElementwiseOpMulI  : I32EnumAttrCase<"MULI", 7, "muli">;
def GPU_ElementwiseOpSUBI  : I32EnumAttrCase<"SUBI", 8, "subi">;
def GPU_ElementwiseOpDivS : I32EnumAttrCase<"DIVS", 9, "divs">;
def GPU_ElementwiseOpDivU : I32EnumAttrCase<"DIVU", 10, "divu">;
def GPU_ElementwiseOpNEGF : I32EnumAttrCase<"NEGATEF", 11, "negatef">;
def GPU_ElementwiseOpNEGS : I32EnumAttrCase<"NEGATES", 12, "negates">;

def MMAElementWise : I32EnumAttr<"MMAElementwiseOp",
  "elementwise operation to apply to mma matrix", [
    GPU_ElementwiseOpAddF,
    GPU_ElementwiseOpMulF,
    GPU_ElementwiseOpSUBF,
    GPU_ElementwiseOpMaxF,
    GPU_ElementwiseOpMinF,
    GPU_ElementwiseOpDivF,
    GPU_ElementwiseOpAddI,
    GPU_ElementwiseOpMulI,
    GPU_ElementwiseOpSUBI,
    GPU_ElementwiseOpDivS,
    GPU_ElementwiseOpDivU,
    GPU_ElementwiseOpNEGF,
    GPU_ElementwiseOpNEGS
  ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}
def MMAElementWiseAttr : EnumAttr<GPU_Dialect, MMAElementWise,
                                  "mma_element_wise">;

def GPU_SubgroupMmaElementwiseOp : GPU_Op<"subgroup_mma_elementwise",
    [Pure,
     AllTypesMatch<["args"]>]>{

  let summary = "GPU warp elementwise operation on a matrix";

  let description = [{
    The `gpu.subgroup_mma_elementwise` takes `!gpu.mma_matrix` inputs and
    compute a new `!gpu.mma_matrix` by applying an elementwise operation to each
    element.

    Since the operation is elementwise and the matrix type must match, the
    matrix elements are processed independently of the matrix layout.

    This op is meant to be used along with `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
     %0 =  %A, %B { opType = "ADD" } :
      (!gpu.mma_matrix<16x16xf16, "COp">, !gpu.mma_matrix<16x16xf16, "COp">)
      -> !gpu.mma_matrix<16x16xf16, "COp">
    ```
  }];

  let arguments = (ins Variadic<GPU_MMAMatrix>:$args,
                       MMAElementWiseAttr:$opType);

  let results = (outs GPU_MMAMatrix:$res);

  let extraClassDeclaration = [{
    gpu::MMAMatrixType getType() {
      return getRes().getType().cast<gpu::MMAMatrixType>();
    }
  }];

  let assemblyFormat = [{
    $opType $args attr-dict `:` functional-type($args, $res)
  }];
}

#endif // GPU_OPS
