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
include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.td"
include "mlir/Dialect/GPU/IR/CompilationAttrs.td"
include "mlir/Dialect/GPU/IR/GPUDeviceMappingAttr.td"
include "mlir/Dialect/GPU/IR/ParallelLoopMapperAttr.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/FunctionInterfaces.td"
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
        Pure,
        DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
        DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>])>,
    Arguments<(ins GPU_DimensionAttr:$dimension,
                   OptionalAttr<IndexAttr>:$upper_bound)>, Results<(outs Index)> {
  let assemblyFormat = "$dimension (`upper_bound` $upper_bound^)? attr-dict";
  let extraClassDefinition = [{
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
      auto dimStr = stringifyDimension(getDimensionAttr().getValue());
      auto opName = getOperationName();
      opName.consume_front("gpu.");
      SmallString<8> resultName({opName, "_", dimStr});
      setNameFn(getResult(),resultName);
    }
  }];
  let builders = [
    OpBuilder<(ins "::mlir::gpu::Dimension":$dimension), [{
      build($_builder, $_state, dimension, /*upperBound=*/nullptr);
    }]>,
    OpBuilder<(ins "::mlir::Type":$resultType, "::mlir::gpu::Dimension":$dimension), [{
      build($_builder, $_state, resultType, dimension, /*upperBound=*/nullptr);
    }]>
  ];
}

def GPU_ClusterDimOp : GPU_IndexOp<"cluster_dim"> {
  let description = [{
    Returns the number of cluster identifiers per grid along
    the x, y, or z `dimension`.

    Example:

    ```mlir
    %cDimX = gpu.cluster_dim x
    ```

    If `upper_bound` is set, then executing (a lowering of) this operation in an
    environment where the clusters per grid is greater than `upper_bound` causes
    undefined behavior.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
  }];
}

def GPU_ClusterDimBlocksOp : GPU_IndexOp<"cluster_dim_blocks"> {
  let description = [{
    Returns the number of thread blocks in the cluster along
    the x, y, or z `dimension`.

    Example:

    ```mlir
    %cDimBlocksX = gpu.cluster_dim_blocks x
    ```

    If `upper_bound` is set, then executing (a lowering of) this operation in an
    environment where the thread blocks per cluster  is greater than `upper_bound`
    causes undefined behavior.

    There is an implicit upper bound of `kMaxClusterDim` (currently 8).
  }];
}

def GPU_ClusterIdOp : GPU_IndexOp<"cluster_id"> {
  let description = [{
    Returns the cluster id, i.e. the index of the current cluster within the
    grid along the x, y, or z `dimension`.

    Example:

    ```mlir
    %cIdY = gpu.cluster_id y
    ```

    If `upper_bound` is set, then executing (a lowering of) this operation in an
    environment where the number of clusters in the grid along `dimension` is
    greater than `upper_bound` causes undefined behavior.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
  }];
}

def GPU_ClusterBlockIdOp : GPU_IndexOp<"cluster_block_id"> {
  let description = [{
    Returns the block id within the cluster along the x, y, or z `dimension`.

    Example:

    ```mlir
    %cBlockIdY = gpu.cluster_block_id y
    ```

    If `upper_bound` is set, then executing (a lowering of) this operation in an
    environment where the number of thread blocks per cluster  along `dimension`
    is greater than `upper_bound` causes undefined behavior.

    There is an implicit upper bound of `kMaxClusterDim` (currently 8).
  }];
}

def GPU_BlockDimOp : GPU_IndexOp<"block_dim"> {
  let description = [{
    Returns the number of threads in the thread block (aka the block size) along
    the x, y, or z `dimension`.

    Example:

    ```mlir
    %bDimX = gpu.block_dim x
    ```

    If `known_block_size` is set on an this operation's enclosing `gpu.func`,
    or `gpu.known_block_size` is set on an enclosing `FunctionOpInterface`
    implementor, or if the enclosing `gpu.launch` specifies a constant size for
    `dimension`'s blocks, these contextual facts may be used to infer that this
    operation has a constant value, though such a transformation will not be
    performed by canonicalization or the default constant folder. Executions which
    cause that constant-value assumption to be false incur undefined behavior.

    If `upper_bound` is set, executions where the bblock size along `dimension`
    exceeds `upper_bound` cause undefined behavior.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
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

    If `upper_bound` is set, or if one can be inferred from `known_grid_size`-type
    annotations in context, executions where the block index in `dimension` would
    be greater than or equal to that bound cause undefined behavior. `upper_bound`
    takes priority over bounds inferrable from context.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
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


    If `known_grid_size` is set on an this operation's enclosing `gpu.func`,
    or `gpu.known_grid_size` is set on an enclosing `FunctionOpInterface`
    implementor, or if the enclosing `gpu.launch` specifies a constant size for
    `dimension`'s grid length, these contextual facts may be used to infer that this
    operation has a constant value, though such a transformation will not be
    performed by canonicalization or the default constant folder. Executions which
    cause that constant-value assumption to be false incur undefined behavior.

    If `upper_bound` is set, executions where the grid size in `dimension` would
    exceed `upper_bound` cause undefined behavior.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
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

    If `upper_bound` is set, or if one can be inferred from `known_block_size`-type
    annotations in context, executions where the thread index would be greater
    than or equal to that bound cause undefined behavior.

    There is an implicit upper bound of `kMaxDim` (currently uint32_t::max).
  }];
}

def GPU_LaneIdOp : GPU_Op<"lane_id", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]> {
  let description = [{
    Returns the lane id within the subgroup (warp/wave).

    Example:
    ```mlir
    %laneId = gpu.lane_id
    ```

    If `upper_bound` is set, executions with more than `upper_bound` lanes per
    subgroup cause undefined behavior. In the abscence of `upper_bound`,
    the lane id is still assumed to be non-negative and less than the
    target-independent `kMaxSubgroupSize` (currently 128).
  }];
  let arguments = (ins OptionalAttr<IndexAttr>:$upper_bound);
  let results = (outs Index:$result);
  let assemblyFormat = "(`upper_bound` $upper_bound^)? attr-dict";
}

def GPU_SubgroupIdOp : GPU_Op<"subgroup_id", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]>,
    Arguments<(ins OptionalAttr<IndexAttr>:$upper_bound)>,
    Results<(outs Index:$result)> {
  let description = [{
    Returns the subgroup id, i.e., the index of the current subgroup within the
    workgroup.

    Example:

    ```mlir
    %sgId = gpu.subgroup_id : index
    ```

    Executions where there are more than `upper_bound` subgroups per workgroup
    cause undefined behavior. There is an implicit upper bound of `kMaxDim`
    (currently uint32_t::max).
  }];

  let assemblyFormat = "(`upper_bound` $upper_bound^)? attr-dict `:` type($result)";
}

def GPU_GlobalIdOp : GPU_IndexOp<"global_id"> {
  let description = [{
    Returns the unique global workitem/thread id, i.e., the unique index of the
    current workitem/thread within all workgroups / grid along the x, y, or z
    `dimension`.

    Example:

    ```mlir
    %gidX = gpu.global_id x
    %gidX = gpu.global_id x upper_bound 65536
    ```

    The `upper_bound` attribute defines an upper bound analogously to the ones on
    `thread_id` and `block_id`. If one is not set, the bound may be inferred from
    a combination of `known_block_size` and `known_grid_size`-type annotations.
  }];
}


def GPU_NumSubgroupsOp : GPU_Op<"num_subgroups", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]>,
    Arguments<(ins OptionalAttr<IndexAttr>:$upper_bound)>,
    Results<(outs Index:$result)> {
  let description = [{
    Returns the number of subgroups within a workgroup.

    Example:

    ```mlir
    %numSg = gpu.num_subgroups : index
    ```

    If `upper_bound` is set, executions with more than `upper_bound` subgroups
    per workgroup cause undefined behavior. There is a default upper bound of
    `kMaxDim` (currently uint32_t::max).
  }];

  let assemblyFormat = "(`upper_bound` $upper_bound^)? attr-dict `:` type($result)";
}

def GPU_SubgroupSizeOp : GPU_Op<"subgroup_size", [
      Pure, DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>]>,
    Arguments<(ins OptionalAttr<IndexAttr>:$upper_bound)>,
    Results<(outs Index:$result)> {
  let description = [{
    Returns the number of threads within a subgroup.

    Example:

    ```mlir
    %sgSz = gpu.subgroup_size : index
    ```

    Executions where the number of threads per subgroup exceed `upper_bound` cause
    undefined behavior. When no `upper_bound` is specified, range analyses and
    similar machinery assume the default bound of `kMaxSubgroupSize`, currently
    128.
  }];

  let assemblyFormat = "(`upper_bound` $upper_bound^)? attr-dict `:` type($result)";
}

def GPU_OptionalDimSizeHintAttr : ConfinedAttr<OptionalAttr<DenseI32ArrayAttr>,
  [AttrConstraint<Or<[IsNullAttr.predicate, DenseArrayCount<3>.predicate]>,
    "with 3 elements (if present)">]>;

def GPU_GPUFuncOp : GPU_Op<"func", [
    HasParent<"GPUModuleOp">, AutomaticAllocationScope, FunctionOpInterface,
    IsolatedFromAbove, AffineScope
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

    A function may optionally be annotated with the block and/or grid sizes
    that will be used when it is launched using the `known_block_size` and
    `known_grid_size` attributes, respectively. If set, these attributes must
    be arrays of three 32-bit integers giving the x, y, and z launch dimensions.
    Launching a kernel that has these annotations, or that calls a function with
    these annotations, using a block size or grid size other than what is specified
    is undefined behavior. These attributes may be set on non-`gpu.func` functions
    by using `gpu.known_block_size` or `gpu.known_grid_size`, but this carries
    the risk that they will de discarded.

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

  let arguments = (ins TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs,
                       OptionalAttr<DictArrayAttr>:$workgroup_attrib_attrs,
                       OptionalAttr<DictArrayAttr>:$private_attrib_attrs,
                       GPU_OptionalDimSizeHintAttr:$known_block_size,
                       GPU_OptionalDimSizeHintAttr:$known_grid_size);
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

    /// Return the index of the first workgroup attribution in the block argument
    /// list.
    unsigned getFirstWorkgroupAttributionIndex() {
      return getFunctionType().getNumInputs();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the workgroup memory
    ArrayRef<BlockArgument> getWorkgroupAttributions() {
      auto begin =
          std::next(getBody().args_begin(), getFirstWorkgroupAttributionIndex());
      auto end = std::next(begin, getNumWorkgroupAttributions());
      return {begin, end};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// workgroup memory.
    BlockArgument addWorkgroupAttribution(Type type, Location loc);

    /// Get the workgroup attribution attribute dictionary for the attribution
    /// at index `index`, counted from the start of the workgroup attributions.
    DictionaryAttr getworkgroupAttributionAttrs(unsigned index);

    /// Set the workgroup attribution attribute dictionary for the attribution
    /// at index `index`, counted from the start of the workgroup attributions.
    void setworkgroupAttributionAttrs(unsigned index, DictionaryAttr value);

    /// Get an attribute for a workgroup attribution. `index` is counted
    /// from the start of the workgroup attributions, not the start of the block.
    Attribute getWorkgroupAttributionAttr(unsigned index, StringAttr name);
    Attribute getWorkgroupAttributionAttr(unsigned index, StringRef name) {
      return getWorkgroupAttributionAttr(index, StringAttr::get((*this)->getContext(), name));
    }

    /// Set an attribute for a workgroup attribution. `index` is counted
    /// from the start of the workgroup attributions, not the start of the block.
    /// A null `value` removes an attributino attribute.
    void setWorkgroupAttributionAttr(unsigned index, StringAttr name, Attribute value);
    void setWorkgroupAttributionAttr(unsigned index, StringRef name, Attribute value) {
      return setWorkgroupAttributionAttr(index, StringAttr::get((*this)->getContext(), name), value);
    }

    /// Returns the number of buffers located in the private memory.
    unsigned getNumPrivateAttributions() {
      return getBody().getNumArguments() - getFunctionType().getNumInputs() -
          getNumWorkgroupAttributions();
    }

    /// Returns the index of the first private buffer in the block argument list.
    unsigned getFirstPrivateAttributionIndex() {
      // Buffers on the private memory always come after buffers on the workgroup
      // memory.
      return getFunctionType().getNumInputs() + getNumWorkgroupAttributions();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the private memory.
    ArrayRef<BlockArgument> getPrivateAttributions() {
      auto begin =
          std::next(getBody().args_begin(), getFirstPrivateAttributionIndex());
      return {begin, getBody().args_end()};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// private memory.
    BlockArgument addPrivateAttribution(Type type, Location loc);

    /// Get the private attribution attribute dictionary for the attribution
    /// at index `index`, counted from the start of the private attributions.
    DictionaryAttr getPrivateAttributionAttrs(unsigned index);

    /// Set the private attribution attribute dictionary for the attribution
    /// at index `index`, counted from the start of the private attributions.
    void setPrivateAttributionAttrs(unsigned index, DictionaryAttr value);

    /// Get an attribute for a private attribution. `index` is counted
    /// from the start of the private attributions, not the start of the block.
    Attribute getPrivateAttributionAttr(unsigned index, StringAttr name);
    Attribute getPrivateAttributionAttr(unsigned index, StringRef name) {
      return getPrivateAttributionAttr(index, StringAttr::get((*this)->getContext(), name));
    }

    /// Set an attribute for a private attribution. `index` is counted
    /// from the start of the private attributions, not the start of the block.
    /// A null `value` removes an attribute.
    void setPrivateAttributionAttr(unsigned index, StringAttr name, Attribute value);
    void setPrivateAttributionAttr(unsigned index, StringRef name, Attribute value) {
      return setPrivateAttributionAttr(index, StringAttr::get((*this)->getContext(), name), value);
    }

    /// Returns the name of the attribute containing the number of buffers
    /// located in the workgroup memory.
    static StringRef getNumWorkgroupAttributionsAttrName() {
      return "workgroup_attributions";
    }

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    Region *getCallableRegion() { return &getBody(); }

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

def GPU_DynamicSharedMemoryOp : GPU_Op<"dynamic_shared_memory", [Pure]>
{
  let summary = "Get the memref for dynamic shared memory";

  let description = [{
    This operation provides a memref pointer to the start of dynamic shared
    memory, often referred to as workgroup memory. It's important to note that
    this dynamic shared memory needs to be allocated at kernel launch. One can
    conveniently utilize `the dynamic_shared_memory_size` parameter of
    `gpu.launch` for this purpose.

    Examples:
    ```mlir
    %0 = gpu.dynamic.shared.memory : memref<?xi8, #gpu.address_space<workgroup>>
    %1 = memref.view %0[%c8192][] : memref<?xi8, #gpu.address_space<workgroup>>
                            to memref<32x64xf32, #gpu.address_space<workgroup>>
    %2 = memref.view %0[%c16384][] : memref<?xi8, #gpu.address_space<workgroup>>
                            to memref<32x64xf32, #gpu.address_space<workgroup>>
    ```
  }];
  let arguments = (ins);
  let results = (outs Arg<MemRefRankOf<[I8], [1]>>:$resultMemref);
  let assemblyFormat = [{ attr-dict `:` type($resultMemref) }];
  let hasVerifier = 1;
}

def LaunchIndx : AnyTypeOf<[Index, I32, I64]>;

def GPU_LaunchFuncOp :GPU_Op<"launch_func", [
      GPU_AsyncOpInterface, AttrSizedOperandSegments,
      AllTypesMatch<["gridSizeX", "gridSizeY", "gridSizeZ", "blockSizeX",
                     "blockSizeY", "blockSizeZ"]>]>,
    Arguments<(ins Variadic<GPU_AsyncToken>:$asyncDependencies,
               SymbolRefAttr:$kernel,
               LaunchIndx:$gridSizeX,
               LaunchIndx:$gridSizeY,
               LaunchIndx:$gridSizeZ,
               LaunchIndx:$blockSizeX,
               LaunchIndx:$blockSizeY,
               LaunchIndx:$blockSizeZ,
               Optional<LaunchIndx>:$clusterSizeX,
               Optional<LaunchIndx>:$clusterSizeY,
               Optional<LaunchIndx>:$clusterSizeZ,
               Optional<I32>:$dynamicSharedMemorySize,
               Variadic<AnyType>:$kernelOperands,
               Optional<AnyType>:$asyncObject)>,
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

    The `gpu.launch_func` also supports kernel launching with clusters if
    supported by the target architecture. The cluster size can be set by
    `clusterSizeX`, `clusterSizeY`, and `clusterSizeZ` arguments. When these
    arguments are present, the Op launches a kernel that clusters the given
    thread blocks. This feature is exclusive to certain architectures.

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

          // (Optional)  Cluster size only for support architectures
          %cIdX = gpu.cluster_id x
          %cIdY = gpu.cluster_id y
          %cIdZ = gpu.cluster_id z

          %cDimX = gpu.cluster_dim x
          %cDimY = gpu.cluster_dim y
          %cDimZ = gpu.cluster_dim z

          "some_op"(%bx, %tx) : (index, index) -> ()
          %42 = load %arg1[%bx] : memref<?xf32, 1>
        }
      }

      %t0 = gpu.wait async
      gpu.launch_func
          async                           // (Optional) Don't block host, return token.
          [%t0]                           // (Optional) Execute only after %t0 has completed.
          @kernels::@kernel_1             // Kernel function.
          clusters in (%cst, %cst, %cst)  // (Optional) Cluster size only for support architectures.
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
      CArg<"ValueRange", "{}">:$asyncDependencies,
      CArg<"std::optional<KernelDim3>", "std::nullopt">:$clusterSize)>,
    OpBuilder<(ins "SymbolRefAttr":$kernel, "KernelDim3":$gridSize,
      "KernelDim3":$blockSize, "Value":$dynamicSharedMemorySize,
      "ValueRange":$kernelOperands,
      "Type":$asyncTokenType,
      CArg<"ValueRange", "{}">:$asyncDependencies,
      CArg<"std::optional<KernelDim3>", "std::nullopt">:$clusterSize)>,
    OpBuilder<(ins "SymbolRefAttr":$kernel, "KernelDim3":$gridSize,
      "KernelDim3":$blockSize, "Value":$dynamicSharedMemorySize,
      "ValueRange":$kernelOperands,
      CArg<"Value", "nullptr">:$asyncObject,
      CArg<"std::optional<KernelDim3>", "std::nullopt">:$clusterSize)>
  ];

  let extraClassDeclaration = [{
    /// The name of the kernel's containing module.
    StringAttr getKernelModuleName();

    /// The name of the kernel.
    StringAttr getKernelName();

    /// Returns true if cluster size is specified.
    bool hasClusterSize() {
      if (getClusterSizeX() && getClusterSizeY() && getClusterSizeZ())
        return true;
      return false;
    }

    /// The number of operands passed to the kernel function.
    unsigned getNumKernelOperands();

    /// The i-th operand passed to the kernel function.
    Value getKernelOperand(unsigned i);

    /// Get the SSA values passed as operands to specify the cluster size.
    /// When the cluster sizes are not specified, it asserts.
    KernelDim3 getClusterSizeOperandValues();

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
      (`<` $asyncObject^ `:` type($asyncObject) `>`)?
      $kernel
      ( `clusters` `in` ` ` `(` $clusterSizeX^ `,` $clusterSizeY `,` $clusterSizeZ `)` )?
      `blocks` `in` ` ` `(` $gridSizeX `,` $gridSizeY `,` $gridSizeZ `)`
      `threads` `in` ` ` `(` $blockSizeX `,` $blockSizeY `,` $blockSizeZ `)`
      custom<LaunchDimType>(type($gridSizeX), ref($clusterSizeX), type($clusterSizeX), type($clusterSizeY), type($clusterSizeZ))
      (`dynamic_shared_memory_size` $dynamicSharedMemorySize^)?
      custom<LaunchFuncOperands>($kernelOperands, type($kernelOperands)) attr-dict
  }];
  let hasVerifier = 1;
}

def GPU_LaunchOp : GPU_Op<"launch", [
      AutomaticAllocationScope, AttrSizedOperandSegments, GPU_AsyncOpInterface,
      DeclareOpInterfaceMethods<InferIntRangeInterface, ["inferResultRanges"]>,
      RecursiveMemoryEffects]>,
    Arguments<(ins Variadic<GPU_AsyncToken>:$asyncDependencies,
               Index:$gridSizeX, Index:$gridSizeY, Index:$gridSizeZ,
               Index:$blockSizeX, Index:$blockSizeY, Index:$blockSizeZ,
               Optional<Index>:$clusterSizeX,
               Optional<Index>:$clusterSizeY,
               Optional<Index>:$clusterSizeZ,
               Optional<I32>:$dynamicSharedMemorySize,
               OptionalAttr<SymbolRefAttr>:$kernelFunc,
               OptionalAttr<SymbolRefAttr>:$kernelModule)>,
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

    The body region has at least _twelve_ arguments, or _eighteen_ if cluster
    dimensions are present, grouped as follows:

    -   three optional arguments that contain cluster identifiers along x,y,z
        dimensions;
    -   three arguments that contain block identifiers along x,y,z dimensions;
    -   three arguments that contain thread identifiers along x,y,z dimensions;
    -   operands of the `gpu.launch` operation as is (i.e. the operands for
        grid and block sizes).
    -   a variadic number of Workgroup memory attributions.
    -   a variadic number of Private memory attributions.

    The `kernelFunc` and `kernelModule` attributes are optional and specifies
    the kernel name and a module in which the kernel should be outlined. 

    Syntax:

    ```
    operation ::= `gpu.launch` (`async` (`[` ssa-id-list `]`)? )?
                             ( `clusters` `(` ssa-id-list `)` `in` ssa-reassignment )?
                             `blocks` `(` ssa-id-list `)` `in` ssa-reassignment
                             `threads` `(` ssa-id-list `)` `in` ssa-reassignment
                             (dynamic_shared_memory_size ssa-use)?
                             memory-attribution
                             region attr-dict?
    ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
    memory-attribution ::= (`workgroup` `(` ssa-id-and-type-list `)`)?
                           (`private` `(` ssa-id-and-type-list `)`)?
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

    // Launch with memory attributions.
    gpu.launch blocks(%bx, %by, %bz) in (%sz_bx = %0, %sz_by = %1, %sz_bz = %2)
               threads(%tx, %ty, %tz) in (%sz_tx = %3, %sz_ty = %4, %sz_tz = %5)
               workgroup(%workgroup: memref<32xf32, 3>)
               private(%private: memref<1xf32, 5>) {
      // Block and thread identifiers, as well as block/grid sizes are
      // immediately usable inside body region.
      "some_op"(%bx, %tx) : (index, index) -> ()
      // Assuming %val1 is defined outside the gpu.launch region.
      %42 = load %workgroup[%bx] : memref<32xf32, 3>
    }

    // Launch with clusters.
    gpu.launch clusters(%cx, %cy, %cz) in (%sz_cx = %0, %sz_cy = %1, %sz_cz = %2)
               blocks(%bx, %by, %bz) in (%sz_bx = %3, %sz_by = %4, %sz_bz = %5)
               threads(%tx, %ty, %tz) in (%sz_tx = %6, %sz_ty = %7, %sz_tz = %8)
    {
      // Cluster, block and thread identifiers, as well as cluster/block/grid
      // sizes are immediately usable inside body region.
      "some_op"(%cx, %bx, %tx) : (index, index, index) -> ()
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
      CArg<"ValueRange", "{}">:$asyncDependencies,
      CArg<"TypeRange", "{}">:$workgroupAttributions,
      CArg<"TypeRange", "{}">:$privateAttributions,
      CArg<"Value", "nullptr">:$clusterSizeX,
      CArg<"Value", "nullptr">:$clusterSizeY,
      CArg<"Value", "nullptr">:$clusterSizeZ)>
  ];

  let extraClassDeclaration = [{
    /// Get the SSA values corresponding to kernel block identifiers.
    KernelDim3 getBlockIds();
    /// Get the SSA values corresponding to kernel thread identifiers.
    KernelDim3 getThreadIds();
    /// Get the SSA values corresponding to kernel cluster identifiers.
    std::optional<KernelDim3> getClusterIds();
    /// Get the SSA values corresponding to kernel grid size.
    KernelDim3 getGridSize();
    /// Get the SSA values corresponding to kernel block size.
    KernelDim3 getBlockSize();
    /// Get the SSA values corresponding to kernel cluster size.
    std::optional<KernelDim3> getClusterSize();

    /// Get the SSA values passed as operands to specify the grid size.
    KernelDim3 getGridSizeOperandValues();
    /// Get the SSA values passed as operands to specify the block size.
    KernelDim3 getBlockSizeOperandValues();
    /// Get the SSA values passed as operands to specify the cluster size.
    std::optional<KernelDim3> getClusterSizeOperandValues();

    static StringRef getBlocksKeyword() { return "blocks"; }
    static StringRef getClustersKeyword() { return "clusters"; }
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

    /// Returns true if cluster size is specified.
    bool hasClusterSize() {
      if (getClusterSizeX() && getClusterSizeY() && getClusterSizeZ())
        return true;
      return false;
    }
    /// Returns the number of operands including cluster size
    unsigned getNumConfigOperands() {
      return kNumConfigOperands + (hasClusterSize() ? 3 : 0);
    }
    /// Returns the number of region attributes including cluster size
    unsigned getNumConfigRegionAttributes() {
      return kNumConfigRegionAttributes + (hasClusterSize() ? 6 : 0);
    }

    /// Returns the keywords used in the custom syntax for this Op.
    static StringRef getWorkgroupKeyword() { return "workgroup"; }
    static StringRef getPrivateKeyword() { return "private"; }

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
          std::next(getBody().args_begin(), getNumConfigRegionAttributes());
      auto end = std::next(begin, getNumWorkgroupAttributions());
      return {begin, end};
    }

    /// Adds a new block argument that corresponds to buffers located in
    /// workgroup memory.
    BlockArgument addWorkgroupAttribution(Type type, Location loc);

    /// Returns the number of buffers located in the private memory.
    unsigned getNumPrivateAttributions() {
      return getBody().getNumArguments() - getNumConfigRegionAttributes() -
          getNumWorkgroupAttributions();
    }

    /// Returns a list of block arguments that correspond to buffers located in
    /// the private memory.
    ArrayRef<BlockArgument> getPrivateAttributions() {
      // Buffers on the private memory always come after buffers on the workgroup
      // memory.
      auto begin =
          std::next(getBody().args_begin(),
                    getNumConfigRegionAttributes() + getNumWorkgroupAttributions());
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
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasRegionVerifier = 1;
  let hasVerifier = 1;
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
    $format attr-dict (`,` $args^ `:` type($args))?
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

def GPU_YieldOp : GPU_Op<"yield", [Pure, ReturnLike, Terminator]>,
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

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>
  ];

  let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
}

// These mirror the reduction combining kinds from the vector dialect.
def GPU_AllReduceOpAdd : I32EnumAttrCase<"ADD", 0, "add">;
def GPU_AllReduceOpMul : I32EnumAttrCase<"MUL", 1, "mul">;
def GPU_AllReduceOpMinUI : I32EnumAttrCase<"MINUI", 2, "minui">;
def GPU_AllReduceOpMinSI : I32EnumAttrCase<"MINSI", 3, "minsi">;
// Follows the `arith.minnumf` semantics.
def GPU_AllReduceOpMinnumF : I32EnumAttrCase<"MINNUMF", 4, "minnumf">;
def GPU_AllReduceOpMaxUI : I32EnumAttrCase<"MAXUI", 5, "maxui">;
def GPU_AllReduceOpMaxSI : I32EnumAttrCase<"MAXSI", 6, "maxsi">;
// Follows the `arith.maxnumf` semantics.
def GPU_AllReduceOpMaxnumF : I32EnumAttrCase<"MAXNUMF", 7, "maxnumf">;
def GPU_AllReduceOpAnd : I32EnumAttrCase<"AND", 8, "and">;
def GPU_AllReduceOpOr  : I32EnumAttrCase<"OR",  9, "or">;
def GPU_AllReduceOpXor : I32EnumAttrCase<"XOR", 10, "xor">;
// Follows the `arith.minimumf` semantics.
def GPU_AllReduceOpMinimumF : I32EnumAttrCase<"MINIMUMF", 11, "minimumf">;
// Follows the `arith.maximumf` semantics.
def GPU_AllReduceOpMaximumF : I32EnumAttrCase<"MAXIMUMF", 12, "maximumf">;

def GPU_AllReduceOperation : I32EnumAttr<"AllReduceOperation",
    "built-in reduction operations supported by gpu.allreduce.",
    [
      GPU_AllReduceOpAdd,
      GPU_AllReduceOpMul,
      GPU_AllReduceOpMinUI,
      GPU_AllReduceOpMinSI,
      GPU_AllReduceOpMinnumF,
      GPU_AllReduceOpMaxUI,
      GPU_AllReduceOpMaxSI,
      GPU_AllReduceOpMaxnumF,
      GPU_AllReduceOpAnd,
      GPU_AllReduceOpOr,
      GPU_AllReduceOpXor,
      GPU_AllReduceOpMinimumF,
      GPU_AllReduceOpMaximumF
    ]>{
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::gpu";
}

def AnyIntegerOrFloat : AnyTypeOf<[AnySignlessInteger, AnyFloat], "Integer or Float">;

def GPU_AllReduceOperationAttr : EnumAttr<GPU_Dialect, GPU_AllReduceOperation,
                                          "all_reduce_op">;

def GPU_AllReduceOp : GPU_Op<"all_reduce",
    [SameOperandsAndResultType, IsolatedFromAbove]> {
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
    accumulation as code region. The reduction operation must be one of:
    *  Integer types: `add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`,
       `or`, `xor`
    *  Floating point types: `add`, `mul`, `minnumf`, `maxnumf`, `minimumf`,
       `maximumf`

    If `uniform` flag is set either none or all work items of a workgroup
    need to execute this op in convergence.
  }];

  let arguments = (ins
    AnyIntegerOrFloat:$value,
    OptionalAttr<GPU_AllReduceOperationAttr>:$op,
    UnitAttr:$uniform
  );
  let results = (outs AnyIntegerOrFloat:$result);

  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{ custom<AllReduceOperation>($op) $value
                          (`uniform` $uniform^)? $body attr-dict
                          `:` functional-type(operands, results) }];

  let hasFolder = 1;
  let hasRegionVerifier = 1;
}

def AnyIntegerOrFloatOr1DVector :
  AnyTypeOf<[AnyIntegerOrFloat, VectorOfRankAndType<[1], [AnyIntegerOrFloat]>]>;

def GPU_SubgroupReduceOp : GPU_Op<"subgroup_reduce", [SameOperandsAndResultType]> {
  let summary = "Reduce values among subgroup.";
  let description = [{
    The `subgroup_reduce` op reduces the values of lanes (work items) across a
    subgroup.

    The subgroup is divided into clusters starting at lane index 0. Within each
    cluster, there are `size` lanes, and the lane index advances by `stride`.
    A reduction is done for each cluster in parallel: every lane in the cluster
    is reduced, and the result is equal for all lanes in the cluster. If `size`
    is omitted, there is a single cluster covering the entire subgroup. If
    `stride` is omitted, the stride is 1 (the cluster's lanes are contiguous).

    When the reduced value is of a vector type, each vector element is reduced
    independently. Only 1-d vector types are allowed.

    Example:

    ```mlir
    %1 = gpu.subgroup_reduce add %a : (f32) -> f32
    %2 = gpu.subgroup_reduce add %b : (vector<4xf16>) -> vector<4xf16>
    %3 = gpu.subgroup_reduce add %c cluster(size = 4) : (f32) -> f32
    %3 = gpu.subgroup_reduce add %c cluster(size = 4, stride = 2) : (f32) -> f32
    ```

    If `uniform` flag is set either none or all lanes of a subgroup need to execute
    this op in convergence.

    The reduction operation must be one of:
    *  Integer types: `add`, `mul`, `minui`, `minsi`, `maxui`, `maxsi`, `and`,
       `or`, `xor`
    *  Floating point types: `add`, `mul`, `minnumf`, `maxnumf`, `minimumf`,
       `maximumf`
  }];

  let arguments = (ins
    AnyIntegerOrFloatOr1DVector:$value,
    GPU_AllReduceOperationAttr:$op,
    UnitAttr:$uniform,
    OptionalAttr<I32Attr>:$cluster_size,
    DefaultValuedAttr<I32Attr,"1">:$cluster_stride
  );
  let results = (outs AnyIntegerOrFloatOr1DVector:$result);

  let builders = [
    OpBuilder<(ins "Value":$value,
               "::mlir::gpu::AllReduceOperation":$op,
               "bool":$uniform), [{
      build($_builder, $_state, value, op, uniform, std::nullopt);
    }]>,
    OpBuilder<(ins "Value":$value,
               "::mlir::gpu::AllReduceOperation":$op,
               "bool":$uniform,
               "std::optional<uint32_t>":$cluster_size), [{
      build($_builder, $_state, value, op, uniform,
            cluster_size ? $_builder.getI32IntegerAttr(*cluster_size) : nullptr);
    }]>,
    OpBuilder<(ins "Value":$value,
               "::mlir::gpu::AllReduceOperation":$op,
               "bool":$uniform,
               "std::optional<uint32_t>":$cluster_size,
               "uint32_t":$cluster_stride), [{
      build($_builder, $_state, value, op, uniform,
            cluster_size ? $_builder.getI32IntegerAttr(*cluster_size) : nullptr,
            cluster_stride);
    }]>
  ];

  let assemblyFormat = [{ custom<AllReduceOperation>($op) $value
                          (`uniform` $uniform^)?
                          (`cluster` `(` `size` `=` $cluster_size^ (`,` `stride` `=` $cluster_stride^)? `)`)?
                          attr-dict
                          `:` functional-type(operands, results) }];

  let hasFolder = 1;
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

def GPU_ShuffleOp : GPU_Op<
    "shuffle", [Pure, AllTypesMatch<["value", "shuffleResult"]>]>,
    Arguments<(ins AnyIntegerOrFloatOr1DVector:$value, I32:$offset, I32:$width,
               GPU_ShuffleModeAttr:$mode)>,
    Results<(outs AnyIntegerOrFloatOr1DVector:$shuffleResult, I1:$valid)> {
  let summary = "Shuffles values within a subgroup.";
  let description = [{
    The "shuffle" op moves values to a across lanes (a.k.a., invocations,
    work items) within the same subgroup. The `width` argument specifies the
    number of lanes that participate in the shuffle, and must be uniform
    across all lanes. Further, the first `width` lanes of the subgroup must
    be active.

    The intepretation of the `offset` arguments depends on the selected
    `mode`.

    Returns the `shuffleResult` and `true` if the current lane id is smaller
    than `width`, and an unspecified value and `false` otherwise.

    `xor` example:

    ```mlir
    %1, %2 = gpu.shuffle xor %0, %offset, %width : f32
    ```

    For lane `k`, returns the value `%0` from lane `k ^ offset`. Every lane
    trades value with exactly one other lane.

    `down` example:

    ```mlir
    %cst1 = arith.constant 1 : i32
    %3, %4 = gpu.shuffle down %0, %cst1, %width : f32
    ```

    For lane `k`, returns the value from lane `(k + 1) % width`.

    `up` example:

    ```mlir
    %cst1 = arith.constant 1 : i32
    %5, %6 = gpu.shuffle up %0, %cst1, %width : f32
    ```

    For lane `k`, returns the value from lane `(k - 1) % width`.

    `idx` example:

    ```mlir
    %cst0 = arith.constant 0 : i32
    %7, %8 = gpu.shuffle idx %0, %cst0, %width : f32
    ```

    Broadcasts the value from lane 0 to all lanes.
  }];

  let assemblyFormat = [{
    $mode $value `,` $offset `,` $width attr-dict `:` type($value)
  }];

  let builders = [
    // Helper function that creates a shuffle with constant offset/width.
    OpBuilder<(ins "Value":$value, "int32_t":$offset, "int32_t":$width,
                   "ShuffleMode":$mode)>
  ];
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
  let hasCanonicalizer = 1;
}

def GPU_GPUModuleOp : GPU_Op<"module", [
      DataLayoutOpInterface, HasDefaultDLTIDataLayout, IsolatedFromAbove,
      NoRegionArguments, SymbolTable, Symbol] # GraphRegionNoTerminator.traits> {
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

    Modules can contain zero or more target attributes. These attributes encode
    how to transform modules into binary strings and are used by the
    `gpu-module-to-binary` pass to transform modules into GPU binaries.

    Modules can contain an optional `OffloadingTranslationAttr` attribute. This
    attribute will be used during the `gpu-module-to-binary` pass to specify the
    `OffloadingTranslationAttr` used when creating the `gpu.binary` operation.

    ```
    gpu.module @symbol_name {
      gpu.func {}
        ...
    }
    // Module with offloading handler and target attributes.
    gpu.module @symbol_name2 <#gpu.select_object<1>> [
        #nvvm.target,
        #rocdl.target<chip = "gfx90a">] {
      gpu.func {}
        ...
    }
    ```
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$name,
                   CArg<"ArrayAttr", "{}">:$targets,
                   CArg<"Attribute", "{}">:$handler)>,
    OpBuilder<(ins "StringRef":$name,
                   "ArrayRef<Attribute>":$targets,
                   CArg<"Attribute", "{}">:$handler)>
  ];

  let arguments = (ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<GPUNonEmptyTargetArrayAttr>:$targets,
      OptionalAttr<OffloadingTranslationAttr>:$offloadingHandler);
  let regions = (region SizedRegion<1>:$bodyRegion);
  let assemblyFormat = [{
    $sym_name
    (`<` $offloadingHandler^ `>`)?
    ($targets^)?
    attr-dict-with-keyword $bodyRegion
  }];

  // We need to ensure the block inside the region is properly terminated;
  // the auto-generated builders do not guarantee that.
  let skipDefaultBuilders = 1;

  let extraClassDeclaration = [{
    /// Checks if `target` is in the `targets` list.
    bool hasTarget(Attribute target);

    /// Sets the targets of the module.
    void setTargets(ArrayRef<TargetAttrInterface> targets);
  }];
}

def GPU_BinaryOp : GPU_Op<"binary", [Symbol]>, Arguments<(ins
      SymbolNameAttr:$sym_name,
      OptionalAttr<OffloadingTranslationAttr>:$offloadingHandler,
      ConfinedAttr<GPUObjectArrayAttr, [ArrayMinCount<1>]>:$objects)
    > {
  let summary = "An Op for storing serialized GPU binary objects.";
  let description = [{
    GPU binaries provide a semantic mechanism for storing GPU objects,
    e.g. the result of compiling a GPU module to an object file.

    This operation has 3 arguments:
     - The name of the binary.
     - An optional attribute implementing the offloading LLVM translation interface.
     - An array of GPU object attributes.

    During translation, the offloading attribute will be called for translating
    GPU `binary` and `launch_func` operations. The default offloading handler is:
    `#gpu.select_object`, this handler selects the first object from the array
    and embeds it as a string.

    Examples:
    ```
      // Selects the first object.
      gpu.binary @myobject [#gpu.object<...>, #gpu.object<...>]
      // Uses the `#foo.my_handler` for handling the binary during translation.
      gpu.binary @myobject <#foo.my_handler> [#gpu.object<...>, #gpu.object<...>]
      // Selects the object with the `#rocdl.target` target attribute.
      gpu.binary @myobject <#gpu.select_object<#rocdl.target>> [#gpu.object<...>, #gpu.object<#rocdl.target, ...>]
    ```
  }];
  let builders = [
    OpBuilder<(ins "StringRef":$name,
                   "Attribute":$offloadingHandler,
                   "ArrayAttr":$objects)>,
    OpBuilder<(ins "StringRef":$name,
                   "Attribute":$offloadingHandler,
                   "ArrayRef<Attribute>":$objects)>
  ];
  let skipDefaultBuilders = 1;
  let assemblyFormat = [{
    $sym_name custom<OffloadingHandler>($offloadingHandler) attr-dict $objects
  }];
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

def GPU_HostUnregisterOp : GPU_Op<"host_unregister">,
    Arguments<(ins AnyUnrankedMemRef:$value)> {
  let summary = "Unregisters a memref for access from device.";
  let description = [{
      This op unmaps the provided host buffer from the device address space.

      This operation may not be supported in every environment, there is not yet a
          way to check at runtime whether this feature is supported.
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
  let results = (outs Res<AnyMemRef, "", [MemAllocAt<0, FullEffect>]>:$memref,
                 Optional<GPU_AsyncToken>:$asyncToken);

  let extraClassDeclaration = [{
    MemRefType getType() { return ::llvm::cast<MemRefType>(getMemref().getType()); }
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
                   Arg<AnyMemRef, "", [MemFreeAt<0, FullEffect>]>:$memref);
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
                   Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$dst,
                   Arg<AnyMemRef, "", [MemReadAt<0, FullEffect>]>:$src);
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
                   Arg<AnyMemRef, "", [MemWriteAt<0, FullEffect>]>:$dst,
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
    row.  If the `transpose` attribute is present then the op does a transposed load.

    For integer types, the resulting `!gpu.mma_matrix` type needs to specify the
    signedness of the data if the matrix type is an `A` or `B` operand for
    `gpu.subgroup_mma_compute`.

    This op is often meant to be used along with `gpu.subgroup_mma_store_matrix` and
    `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
     %0 = gpu.subgroup_mma_load_matrix src[%i,%j] : {leadDimension = 32 : i32}
          : memref<32x32xf16, 3>, !gpu.mma_matrix<16x16xf16, "AOp">
    ```
  }];

  let arguments = (ins Arg<GPU_MMAMemRef, "",
                          [MemReadAt<0, FullEffect>]>:$srcMemref,
                  Variadic<Index>:$indices,
                  IndexAttr:$leadDimension,
                  OptionalAttr<UnitAttr>:$transpose);

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
    specifies the leading dimension of the destination matrix. If the
    `transpose` attribute is present then the op does a transposed store.

    This op is often meant to be used along with `gpu.subgroup_mma_load_matrix` and
    `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
    gpu.subgroup_mma_store_matrix %D, %sg[%i,%j] : { leadDimension = 32 : i32}
                    : !gpu.mma_matrix<16x16xf16, "COp">, memref<32x32xf16, 3>
    ```
  }];

  let arguments = (ins Arg<MMAMatrixOf<[SI8, UI8, I32, F16, F32]>>:$src,
                  Arg<GPU_MMAMemRef, "",[MemWriteAt<0, FullEffect>]>:$dstMemref,
                  Variadic<Index>:$indices,
                  IndexAttr:$leadDimension,
                  OptionalAttr<UnitAttr>:$transpose);

  let assemblyFormat = [{
    $src`,` $dstMemref`[`$indices`]` attr-dict `:` type($src)`,` type($dstMemref)
  }];
  let hasVerifier = 1;
}

def GPU_SubgroupMmaComputeOp
    : GPU_Op<"subgroup_mma_compute", [Pure, AllTypesMatch<["opC", "res"]>]> {

  let summary = "GPU warp synchronous matrix multiply accumulate";

  let description = [{
    The `gpu.subgroup_mma_compute` operation performs a matrix-multiply accumulate (mma)
    operation using all the threads in a subgroup.

    This operation takes three `!gpu.mma_matrix`s as arguments: these hold `A`,
    `B` and `C`operands for the mma operation. The operation performed is represented
    as `C += A * B`. The op returns a `!gpu.mma_matrix` which contains the result of
    the operation held by all threads in a subgroup. `a_transpose` or
    `b_transpose` if present, signify that the respective operand was loaded in a
    transposed manner. The transpose operands are required to map to correct
    underlying intrisics but they currently do not seem to affect correctness
    even if they are absent given that the operands were loaded correctly using
    the `transpose` attribute in `gpu.subgroup_mma_load_matrix` op.

    For integer types, the `A` and `B` matrices carry their signedness with their
    types. The accumulator type is expected to be signless and imply a signed integer
    with a greater width than the other two operands.

    This op is meant to be used along with `gpu.subgroup_mma_store_matrix` and
    `gpu.subgroup_mma_load_matrix` ops.

    Example:

    ```mlir
    %D = gpu.subgroup_mma_compute_matrix %A, %B, %C :
      !gpu.mma_matrix<16x16xf16, "AOp">, !gpu.mma_matrix<16x16xf16, "BOp">>
      -> !gpu.mma_matrix<16x16xf16, "COp">
    ```
  }];

  let arguments = (ins Arg<MMAMatrixOf<[SI8, UI8, F16, F32]>>:$opA,
                  Arg<MMAMatrixOf<[SI8, UI8, F16, F32]>>:$opB,
                  Arg<MMAMatrixOf<[I32, F16, F32]>>:$opC,
                  OptionalAttr<UnitAttr>:$a_transpose,
                  OptionalAttr<UnitAttr>:$b_transpose);

  let results = (outs GPU_MMAMatrix : $res);

  let assemblyFormat = [{
    $opA`,` $opB`,` $opC attr-dict `:` type($opA)`,` type($opB) `->` type($res)
  }];
  let hasVerifier = 1;
}

def GPU_SubgroupMmaConstantMatrixOp : GPU_Op<"subgroup_mma_constant_matrix",
    [Pure,
     TypesMatchWith<"value type matches element type of mma_matrix",
                    "res", "value",
                    "::llvm::cast<gpu::MMAMatrixType>($_self).getElementType()">]>{

  let summary = "GPU warp synchronous constant matrix";

  let description = [{
    The `gpu.subgroup_mma_constant_matrix` creates a `!gpu.mma_matrix` with
    constant elements.

    The operation takes a scalar input and return a `!gpu.mma_matrix` where
    each element of is equal to the operand constant. The destination
    mma_matrix type must have elememt type equal to the constant type. Since
    the layout of `!gpu.mma_matrix` is opaque this only support setting all the
    elements to the same value.

    This op is meant to be used along with `gpu.subgroup_mma_compute`.

    Example:

    ```mlir
     %0 = gpu.subgroup_mma_constant_matrix %a :
       !gpu.mma_matrix<16x16xf16, "AOp">
     %1 = gpu.subgroup_mma_constant_matrix %b :
       !gpu.mma_matrix<16x16xf32, "COp">
    ```
  }];

  let arguments = (ins AnyTypeOf<[SI8, UI8, I32, F16, F32]>:$value);

  let results = (outs GPU_MMAMatrix:$res);

  let extraClassDeclaration = [{
    gpu::MMAMatrixType getType() {
      return ::llvm::cast<gpu::MMAMatrixType>(getRes().getType());
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
def GPU_ElementwiseOpEXTF : I32EnumAttrCase<"EXTF", 13, "extf">;

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
    GPU_ElementwiseOpNEGS,
    GPU_ElementwiseOpEXTF
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
      return ::llvm::cast<gpu::MMAMatrixType>(getRes().getType());
    }
  }];

  let assemblyFormat = [{
    $opType $args attr-dict `:` functional-type($args, $res)
  }];
}

//
// Operation on sparse matrices, called from the host
// (currently lowers to cuSparse for CUDA only, no ROCM lowering).
//

def GPU_CreateDnTensorOp : GPU_Op<"create_dn_tensor", [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {
  let summary = "Create dense tensor operation";
  let description = [{
    The `gpu.create_dn_tensor` operation initializes a dense tensor from
    the given values buffer and sizes. The buffer must already be copied
    from the host to the device prior to using this operation. The
    operation returns a handle to the dense tensor descriptor.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %dmat, %token = gpu.create_dn_tensor async [%dep] %mem, %dims : index, index into memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       AnyMemRef:$memref,
                       Variadic<Index>:$dims);
  let results = (outs Res<GPU_SparseDnTensorHandle>:$dnTensor, Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $memref `,` $dims attr-dict `:` type($dims) `into` type($memref)
  }];
}

def GPU_DestroyDnTensorOp : GPU_Op<"destroy_dn_tensor", [GPU_AsyncOpInterface]> {
  let summary = "Destroy dense tensor operation";
  let description = [{
    The `gpu.destroy_dn_tensor` operation releases all resources of a dense
    tensor represented by a handle that was previously created by a
    `gpu.create_dn_tensor` operation.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %token = gpu.destroy_dn_tensor async [%dep] %dnTensor
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       Arg<GPU_SparseDnTensorHandle>:$dnTensor);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $dnTensor attr-dict
  }];
}

def GPU_CreateCooOp : GPU_Op<"create_coo", [GPU_AsyncOpInterface]> {
  let summary = "Create sparse matrix in COO format operation";
  let description = [{
    The `gpu.create_coo` operation initializes a sparse matrix in COO format
    with the given sizes from the given index and values buffers. The buffers
    must already be copied from the host to the device prior to using this
    operation. The operation returns a handle to the sparse matrix descriptor.
    Note that this operation builds the COO in SoA format.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %spmat, %token = gpu.create_coo async [%dep] %rows, %cols, %nnz, %rowIdx,
        %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       Index:$rows,
                       Index:$cols,
                       Index:$nnz,
                       AnyMemRef:$rowIdxs,
                       AnyMemRef:$colIdxs,
                       AnyMemRef:$values);
  let results = (outs Res<GPU_SparseSpMatHandle>:$spmat,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $rows `,` $cols `,` $nnz `,` $rowIdxs `,` $colIdxs `,` $values attr-dict
    `:` type($rowIdxs) `,` type($colIdxs) `,` type($values)
  }];
}

def GPU_CreateCooAoSOp : GPU_Op<"create_coo_aos", [GPU_AsyncOpInterface]> {
  let summary = "Create sparse matrix in COO format operation (AoS)";
  let description = [{
    The `gpu.create_coo_aos` operation initializes a sparse matrix in COO format
    with the given sizes from the given index and values buffers. The buffers
    must already be copied from the host to the device prior to using this
    operation. The operation returns a handle to the sparse matrix descriptor.
    Unlike the default `gpu.create_coo` operation, this operation builds the
    COO format from a single index buffer in AoS format (note that this
    feature has been deprecated in cuSparse 11.2).

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %spmat, %token = gpu.create_coo_aos async [%dep] %rows, %cols, %nnz, %idxs,
        %values : memref<?xindex>, memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Index:$rows,
                   Index:$cols,
                   Index:$nnz,
                   AnyMemRef:$idxs,
                   AnyMemRef:$values);
  let results = (outs Res<GPU_SparseSpMatHandle>:$spmat,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $rows `,` $cols `,` $nnz `,` $idxs `,` $values attr-dict
    `:` type($idxs) `,` type($values)
  }];
}

def GPU_CreateCsrOp : GPU_Op<"create_csr", [GPU_AsyncOpInterface]> {
  let summary = "Create sparse matrix in CSR format operation";
  let description = [{
    The `gpu.create_csr` operation initializes a sparse matrix in CSR format
    with the given sizes from the given position, index, and values buffers.
    The buffers must already be copied from the host to the device prior to
    using this operation. The operation returns a handle to the sparse
    matrix descriptor.

    The CSR format has exactly the same memory layout as its transpose
    in CSC format (and vice versa).

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %spmat, %token = gpu.create_csr async [%dep] %rows, %cols, %nnz, %rowPos,
        %colIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Index:$rows,
                   Index:$cols,
                   Index:$nnz,
                   AnyMemRef:$rowPos,
                   AnyMemRef:$colIdxs,
                   AnyMemRef:$values);
  let results = (outs Res<GPU_SparseSpMatHandle>:$spmat,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $rows `,` $cols `,` $nnz `,` $rowPos `,` $colIdxs `,` $values attr-dict
    `:` type($rowPos) `,` type($colIdxs) `,` type($values)
  }];
}

def GPU_CreateCscOp : GPU_Op<"create_csc", [GPU_AsyncOpInterface]> {
  let summary = "Create sparse matrix in CSC format operation";
  let description = [{
    The `gpu.create_csc` operation initializes a sparse matrix in CSC format
    with the given sizes from the given position, index, and values buffers.
    The buffers must already be copied from the host to the device prior to
    using this operation. The operation returns a handle to the sparse
    matrix descriptor.

    The CSC format has exactly the same memory layout as its transpose
    in CSR format (and vice versa).

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %spmat, %token = gpu.create_csc async [%dep] %rows, %cols, %nnz, %colPos,
        %rowIdx, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Index:$rows,
                   Index:$cols,
                   Index:$nnz,
                   AnyMemRef:$colPos,
                   AnyMemRef:$rowIdxs,
                   AnyMemRef:$values);
  let results = (outs Res<GPU_SparseSpMatHandle>:$spmat,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $rows `,` $cols `,` $nnz `,` $colPos `,` $rowIdxs `,` $values attr-dict
    `:` type($colPos) `,` type($rowIdxs) `,` type($values)
  }];
}

def GPU_CreateBsrOp : GPU_Op<"create_bsr", [GPU_AsyncOpInterface]> {
  let summary = "Create sparse matrix in BSR format operation";
  let description = [{
    The `gpu.create_bsr` operation initializes a sparse matrix in BSR format
    with the given sizes for the matrix and blocks from the given position,
    index, and values buffers. The buffers must already be copied from the
    host to the device prior to using this operation. The operation returns
    a handle to the sparse matrix descriptor.

    The BSR format is similar to CSR, where the column indices represent
    two-dimensional blocks instead of a single matrix entry. Note that this
    operation (currently) only supports storage with **square** blocks,
    i.e., `rBlockSize == cBlockSize`.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %spmat, %token = gpu.create_bsr async [%dep]
       %brows, %bcols, %bnnz, %rBlockSize, %cBlockSize,
       %bRowPos, %bColIdxs, %values : memref<?xindex>, memref<?xindex>, memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   Index:$brows,
                   Index:$bcols,
                   Index:$bnnz,
                   Index:$rBlockSize,
                   Index:$cBlockSize,
                   AnyMemRef:$bRowPos,
                   AnyMemRef:$bColIdxs,
                   AnyMemRef:$values);
  let results = (outs Res<GPU_SparseSpMatHandle>:$spmat,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $brows `,` $bcols `,` $bnnz `,` $rBlockSize `,` $cBlockSize `,`
    $bRowPos `,` $bColIdxs `,` $values attr-dict
    `:` type($bRowPos) `,` type($bColIdxs) `,` type($values)
  }];
}

def GPU_Prune2To4SpMatFlag : I32EnumAttr<"Prune2To4SpMatFlag",
  "pruning strategy for 2:4 sparse matrix",
  [
    I32EnumAttrCase<"NONE", 0>,
    I32EnumAttrCase<"PRUNE_ONLY", 1>,
    I32EnumAttrCase<"PRUNE_AND_CHECK", 2>,
  ]> {
    let genSpecializedAttr = 0;
    let cppNamespace = GPU_Dialect.cppNamespace;
}

def GPU_Prune2To4SpMatFlagAttr : EnumAttr<GPU_Dialect, GPU_Prune2To4SpMatFlag,
                                   "prune_2to4_spmat_flag">{
  let defaultValue = "Prune2To4SpMatFlag::PRUNE_AND_CHECK";
}


def GPU_Create2To4SpMatOp : GPU_Op<"create_2to4_spmat", [GPU_AsyncOpInterface]> {
  let summary = "Create sparse matrix with 2:4 sparsity operation";
  let description = [{
    The `gpu.create_2to4_spmat` operation initializes a sparse matrix in dense
    format with 2:4 sparsity.
    The buffers must already be copied from the host to the device prior to
    using this operation. The operation returns a handle to the sparse
    matrix descriptor.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %spmat, %token = gpu.create_2to4_spmat async [%dep] {PRUNE_AND_CHECK} %rows, %cols, %mem: memref<?xf64>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       Index:$rows,
                       Index:$cols,
                       GPU_Prune2To4SpMatFlagAttr:$pruneFlag,
                       AnyMemRef:$memref);
  let results = (outs Res<GPU_SparseSpMatHandle>:$spMat,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
     `{` $pruneFlag `}` $rows `,` $cols `,` $memref attr-dict `:` type($memref)
  }];
}

def GPU_DestroySpMatOp : GPU_Op<"destroy_sp_mat", [GPU_AsyncOpInterface]> {
  let summary = "Destroy sparse matrix operation";
  let description = [{
    The `gpu.destroy_sp_mat` operation releases all resources of a sparse
    matrix represented by a handle that was previously created by a
    one of the sparse matrix creation operations.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %token = gpu.destroy_sp_mat async [%dep] %spmat
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       Arg<GPU_SparseSpMatHandle>:$spmat);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies) $spmat attr-dict
  }];
}

// To avoid coupling this dialect with cusparse.h specifics, we hardcoded magic
// literals in this enum. Note that this should be kept in sync with
// cusparseOperation_t in cusparse.h:
// typedef enum {
// CUSPARSE_OPERATION_NON_TRANSPOSE       = 0,
// CUSPARSE_OPERATION_TRANSPOSE           = 1,
// CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
// } cusparseOperation_t;
// TODO: find a proper way to keep them in sync?
def GPU_TransposeMode : I32EnumAttr<"TransposeMode",
    "transpose mode of sparse matrix supported by sparse tensor ops",
    [
      I32EnumAttrCase<"NON_TRANSPOSE", 0>,
      I32EnumAttrCase<"TRANSPOSE", 1>,
      I32EnumAttrCase<"CONJUGATE_TRANSPOSE", 2>,
    ]> {
      let genSpecializedAttr = 0;
      let cppNamespace = GPU_Dialect.cppNamespace;
}

def GPU_TransposeModeAttr : EnumAttr<GPU_Dialect, GPU_TransposeMode,
                                   "mat_transpose_mode">{
  let defaultValue = "TransposeMode::NON_TRANSPOSE";
}

def GPU_SpMVBufferSizeOp : GPU_Op<"spmv_buffer_size", [GPU_AsyncOpInterface]> {
  let summary = "Precompute buffersize for SpMV operation";
  let description = [{
    The `gpu.spmv_buffer_size` operation returns the buffer size required
    to perform the SpMV operation on the given sparse matrix and dense vectors.
    The operation expects handles returned by previous sparse operations
    to construct an environment and the operands for SpMV.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.

    Example:

    ```mlir
    %buffersz, %token = gpu.spmv_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY into f32
    ```
  }];
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_TransposeModeAttr:$modeA,
                       GPU_SparseSpMatHandle:$spmatA,
                       GPU_SparseDnTensorHandle:$dnX,
                       GPU_SparseDnTensorHandle:$dnY,
                       TypeAttr:$computeType);
  let results = (outs Res<Index>:$bufferSz,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
      "Type":$bufferSz,
      "Type":$asyncToken,
      "ValueRange":$asyncDependencies,
      "Value":$spmatA,
      "Value":$dnX,
      "Value":$dnY,
      "Type":$computeType)
      , [{
    auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
    return build($_builder, $_state, bufferSz, asyncToken, asyncDependencies,
                 modeA, spmatA, dnX, dnY, computeType);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $spmatA (`{` $modeA^ `}`)? `,` $dnX `,` $dnY attr-dict  `into` $computeType
  }];
}

def GPU_SpMVOp : GPU_Op<"spmv", [GPU_AsyncOpInterface]> {
  let summary = "SpMV operation";
  let description = [{
    The `gpu.spmv` operation performs the SpMV operation on the given sparse matrix,
    dense vectors, and buffer.  The operation expects handles returned by previous
    sparse operations to construct an environment and the operands for SpMV. The
    buffer must have been allocated on the device.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.

    Example:

    ```mlir
    %token = gpu.spmv async [%dep] %spmatA{TRANSPOSE}, %dnX, %dnY : memref<?xf64> into bf16
    ```
  }];
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_TransposeModeAttr:$modeA,
                       GPU_SparseSpMatHandle:$spmatA,
                       GPU_SparseDnTensorHandle:$dnX,
                       GPU_SparseDnTensorHandle:$dnY,
                       TypeAttr:$computeType,
                       AnyMemRef:$buffer);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
      "Type":$asyncToken,
      "ValueRange":$asyncDependencies,
      "Value":$spmatA,
      "Value":$dnX,
      "Value":$dnY,
      "Type":$computeType,
      "Value":$buffer), [{
    auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
    return build($_builder, $_state, asyncToken, asyncDependencies, modeA,
                 spmatA, dnX, dnY, computeType, buffer);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $spmatA (`{` $modeA^ `}`)? `,` $dnX `,` $dnY `,` $buffer attr-dict `:` type($buffer) `into` $computeType
  }];
}

def GPU_SpMMBufferSizeOp : GPU_Op<"spmm_buffer_size", [GPU_AsyncOpInterface, AttrSizedResultSegments]> {
  let summary = "Precompute buffersize for SpMM operation";
  let description = [{
    The `gpu.spmm_buffer_size` operation returns the buffer size required
    to perform the SpMM operation on the given sparse and dense matrix.
    The operation expects handles returned by previous sparse operations
    to construct an environment and the operands for SpMM.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.

    Example:

    ```mlir
    %bufferszs, %token = gpu.spmm_buffer_size async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC : i64 into f32
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_TransposeModeAttr:$modeA,
                       GPU_TransposeModeAttr:$modeB,
                       GPU_SparseSpMatHandle:$spmatA,
                       GPU_SparseDnTensorHandle:$dnmatB,
                       GPU_SparseDnTensorHandle:$dnmatC,
                       TypeAttr:$computeType);
  let results = (outs Variadic<Index>:$bufferSzs,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
      "Type":$bufferSzs,
      "Type":$asyncToken,
      "ValueRange":$asyncDependencies,
      "Value":$spmatA,
      "Value":$dnmatB,
      "Value":$dnmatC,
      "Type":$computeType), [{
    auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
    auto modeB = gpu::TransposeMode::NON_TRANSPOSE;
    return build($_builder, $_state, bufferSzs, asyncToken, asyncDependencies,
                 modeA, modeB, spmatA, dnmatB, dnmatC, computeType);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $spmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $dnmatC attr-dict `:` type($bufferSzs) `into` $computeType
  }];
}

def GPU_SpMMOp : GPU_Op<"spmm", [GPU_AsyncOpInterface, AttrSizedOperandSegments]> {
  let summary = "SpMM operation";
  let description = [{
    The `gpu.spmm` operation performs the SpMM operation on the given sparse and
    dense matrix, and buffer.  The operation expects handles returned by previous
    sparse operations to construct an environment and the operands for SpMM. The
    buffer must have been allocated on the device.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.

    Example:

    ```mlir
    %token = gpu.spmm async [%dep] %spmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %dnmatC, %buffers : type($buffers) into f32
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_TransposeModeAttr:$modeA,
                       GPU_TransposeModeAttr:$modeB,
                       GPU_SparseSpMatHandle:$spmatA,
                       GPU_SparseDnTensorHandle:$dnmatB,
                       GPU_SparseDnTensorHandle:$dnmatC,
                       TypeAttr:$computeType,
                       Variadic<AnyMemRef>:$buffers);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
      "Type":$asyncToken,
      "ValueRange":$asyncDependencies,
      "Value":$spmatA,
      "Value":$dnmatB,
      "Value":$dnmatC,
      "Type":$computeType,
      "ValueRange":$buffers), [{
    auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
    auto modeB = gpu::TransposeMode::NON_TRANSPOSE;
    return build($_builder, $_state, asyncToken, asyncDependencies, modeA,
                 modeB, spmatA, dnmatB, dnmatC, computeType, buffers);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $spmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $dnmatC `,` $buffers attr-dict `:` type($buffers) `into` $computeType
  }];
}

def GPU_SDDMMBufferSizeOp : GPU_Op<"sddmm_buffer_size", [GPU_AsyncOpInterface]> {
  let summary = "Precompute buffersize for SDDMM operation";
  let description = [{
    The `gpu.sddmm_buffer_size` operation returns the buffer size required
    to perform the SDDMM operation on the given sparse and dense matrices.
    The operation expects handles returned by previous sparse operations
    to construct an environment and the operands for SDDMM.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %buffersz, %token = gpu.sddmm_buffer_size async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC into f32
    ```

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPU_TransposeModeAttr:$modeA,
                   GPU_TransposeModeAttr:$modeB,
                   GPU_SparseDnTensorHandle:$dnmatA,
                   GPU_SparseDnTensorHandle:$dnmatB,
                   GPU_SparseSpMatHandle:$spmatC,
                   TypeAttr:$computeType);
  let results = (outs Res<Index>:$bufferSz, Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
      "Type":$bufferSz,
      "Type":$asyncToken,
      "ValueRange":$asyncDependencies,
      "Value":$dnmatA,
      "Value":$dnmatB,
      "Value":$spmatC,
      "Type":$computeType), [{
    auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
    auto modeB = gpu::TransposeMode::NON_TRANSPOSE;
    return build($_builder, $_state, bufferSz, asyncToken, asyncDependencies,
                 modeA, modeB, dnmatA, dnmatB, spmatC, computeType);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $dnmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $spmatC attr-dict `into` $computeType
  }];
}

def GPU_SDDMMOp : GPU_Op<"sddmm", [GPU_AsyncOpInterface]> {
  let summary = "SDDMM operation";
  let description = [{
    The `gpu.sddmm` operation performs the SDDMM operation on the given sparse and
    dense matrices, and buffer.  The operation expects handles returned by previous
    sparse operations to construct an environment and the operands for SDDMM. The
    buffer must have been allocated on the device.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a !gpu.async.token in addition to the environment.

    Example:

    ```mlir
    %token = gpu.sddmm async [%dep] %dnmatA{TRANSPOSE}, %dnmatB{TRANSPOSE}, %spmatC, %buffer into f32
    ```

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                   GPU_TransposeModeAttr:$modeA,
                   GPU_TransposeModeAttr:$modeB,
                   GPU_SparseDnTensorHandle:$dnmatA,
                   GPU_SparseDnTensorHandle:$dnmatB,
                   GPU_SparseSpMatHandle:$spmatC,
                   TypeAttr:$computeType,
                   AnyMemRef:$buffer);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
    "Type":$asyncToken,
    "ValueRange":$asyncDependencies,
    "Value":$dnmatA,
    "Value":$dnmatB,
    "Value":$spmatC,
    "Type":$computeType,
    "Value":$buffer), [{
  auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
  auto modeB = gpu::TransposeMode::NON_TRANSPOSE;
  return build($_builder, $_state, asyncToken, asyncDependencies, modeA,
                modeB, dnmatA, dnmatB, spmatC, computeType, buffer);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $dnmatA (`{` $modeA^ `}`)? `,` $dnmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $buffer attr-dict `:` type($buffer) `into` $computeType
  }];
}

def GPU_SpGEMMWorkEstimationOrComputeKind : I32EnumAttr<"SpGEMMWorkEstimationOrComputeKind",
    "choose whether spgemm_work_estimation_or_compute does work estimation or compute",
    [
      I32EnumAttrCase<"WORK_ESTIMATION", 0>,
      I32EnumAttrCase<"COMPUTE", 1>,
    ]> {
      let genSpecializedAttr = 0;
      let cppNamespace = GPU_Dialect.cppNamespace;
}

def GPU_SpGEMMWorkEstimationOrComputeKindAttr : EnumAttr<GPU_Dialect,
    GPU_SpGEMMWorkEstimationOrComputeKind,
    "spgemm_work_estimation_or_compute_kind"> {}

def GPU_SpGEMMCreateDescrOp : GPU_Op<"spgemm_create_descr", [GPU_AsyncOpInterface]> {
  let summary = "SpGEMM Create Descr operation";
  let description = [{
    The `gpu.spgemm_create_descr` creates a descriptor for the SpGEMM operation.
    The descriptor describes the SpGEMM operation and stores the internal data
    throughout the computation. It needs to be passed as an argument to
    spgemm_* operations.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a `!gpu.async.token` in addition to the environment.

    Example:

    ```mlir
    %desc, %token = gpu.spgemm_create_descr async [%dep]
    ```
  }];
  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies);
  let results = (outs GPU_SparseSpGEMMOpHandle:$desc,
                      Optional<GPU_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    attr-dict
  }];
}

def GPU_SpGEMMDestroyDescrOp : GPU_Op<"spgemm_destroy_descr", [GPU_AsyncOpInterface]> {
  let summary = "SpGEMM Destroy Descr operation";
  let description = [{
    The `gpu.spgemm_destroy_descr` destroys the SpGEMM operation descriptor.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a `!gpu.async.token` in addition to the environment.

    Example:

    ```mlir
    %token = gpu.spgemm_destroy_descr async [%dep] %desc
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_SparseSpGEMMOpHandle:$desc);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);
  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $desc attr-dict
  }];
}

def GPU_SpGEMMWorkEstimationOrComputeOp : GPU_Op<"spgemm_work_estimation_or_compute", [GPU_AsyncOpInterface]> {
  let summary = "SpGEMM work estimation operation";
  let description = [{
    The `gpu.spgemm_work_estimation_or_compute` is used to call
    cusparseSpGEMM_workEstimation or cusparseSpGEMM_compute. Both of them are
    for both determining the buffer size and performing the actual computation.
    The operation expects handles returned by previous sparse operations to
    construct an environment and the operands for SpGEMM.
    The buffer must have been allocated on the device.

    C' = alpha * op(A) * op(B) + beta * C

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a `!gpu.async.token` in addition to the environment.

    Example:

    ```mlir
    %bufferSz, %token = gpu.spgemm_work_estimation_or_compute async [%dep] {COMPUTE}
                          %desc, %spmatA{NON_TRANSPOSE}, %spmatB{NON_TRANSPOSE},
                          %spmatC, %spgemmDesc, %c0, %alloc: f32 into
                          memref<0xi8>
    ```

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_SparseSpGEMMOpHandle:$desc,
                       GPU_TransposeModeAttr:$modeA,
                       GPU_TransposeModeAttr:$modeB,
                       GPU_SparseSpMatHandle:$spmatA,
                       GPU_SparseSpMatHandle:$spmatB,
                       GPU_SparseSpMatHandle:$spmatC,
                       TypeAttr:$computeType,
                       Index:$bufferSz,
                       AnyMemRef:$buffer,
                       GPU_SpGEMMWorkEstimationOrComputeKindAttr:$kind);
  let results = (outs Res<Index>:$bufferSzNew,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
    "Type":$bufferSzNew,
    "Type":$asyncToken,
    "ValueRange":$asyncDependencies,
    "Value":$desc,
    "Value":$spmatA,
    "Value":$spmatB,
    "Value":$spmatC,
    "Type":$computeType,
    "Value":$bufferSz,
    "Value":$buffer), [{
  auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
  auto modeB = gpu::TransposeMode::NON_TRANSPOSE;
  auto kind = gpu::SpGEMMWorkEstimationOrComputeKind::WORK_ESTIMATION;
  return build($_builder, $_state, bufferSzNew, asyncToken, asyncDependencies, desc,
               modeA, modeB, spmatA, spmatB, spmatC, computeType, bufferSz, buffer, kind);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    `{` $kind `}` $spmatA (`{` $modeA^ `}`)? `,` $spmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $desc `,` $bufferSz `,` $buffer  attr-dict `:` $computeType `into` type($buffer)
  }];
}

def GPU_SpGEMMCopyOp : GPU_Op<"spgemm_copy", [GPU_AsyncOpInterface]> {
  let summary = "SpGEMM copy operation";
  let description = [{
    The `gpu.spgemm_copy` operation copies the sparse matrix result of
    a SpGEMM computation.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a `!gpu.async.token` in addition to the environment.

    Example:

    ```mlir
    gpu.spgemm_copy %spmatA, %spmatB, %spmatC, %spgemmDesc: f32
    ```

    The matrix arguments can also be associated with one of the following
    operators: NON_TRANSPOSE, TRANSPOSE, CONJUGATE_TRANSPOSE. The default value
    is NON_TRANSPOSE.
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_SparseSpGEMMOpHandle:$desc,
                       GPU_TransposeModeAttr:$modeA,
                       GPU_TransposeModeAttr:$modeB,
                       GPU_SparseSpMatHandle:$spmatA,
                       GPU_SparseSpMatHandle:$spmatB,
                       GPU_SparseSpMatHandle:$spmatC,
                       TypeAttr:$computeType);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let builders = [OpBuilder<(ins
    "Type":$asyncToken,
    "ValueRange":$asyncDependencies,
    "Value":$desc,
    "Value":$spmatA,
    "Value":$spmatB,
    "Value":$spmatC,
    "Type":$computeType), [{
  auto modeA = gpu::TransposeMode::NON_TRANSPOSE;
  auto modeB = gpu::TransposeMode::NON_TRANSPOSE;
  return build($_builder, $_state, asyncToken, asyncDependencies, desc,
               modeA, modeB, spmatA, spmatB, spmatC, computeType);}]>
  ];

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $spmatA (`{` $modeA^ `}`)? `,` $spmatB (`{` $modeB^ `}`)? `,` $spmatC `,` $desc attr-dict `:` $computeType
  }];
}

def GPU_SpMatGetSizeOp : GPU_Op<"spmat_get_size", [GPU_AsyncOpInterface]> {
  let summary = "SpMat get size operation";
  let description = [{
    The `gpu.spmat_get_size` operation retrieves the number of rows, number of
    columns, and number of non-zero elements of a sparse matrix.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a `!gpu.async.token` in addition to the environment.

    Example:

    ```mlir
    %rows, %cols, %nnz, %token = gpu.spmat_get_size async [%dep] %spmatC
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       GPU_SparseSpMatHandle:$spmat);
  let results = (outs Index:$rows,
                      Index:$cols,
                      Index:$nnz,
                      Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
    $spmat attr-dict
  }];
}

def GPU_SetCsrPointersOp : GPU_Op<"set_csr_pointers", [GPU_AsyncOpInterface]> {
  let summary = "SpGEMM get size operation";
  let description = [{
    The `gpu.set_csr_pointers` assigns the given positions, coordinates,
    and values buffer that reside on the device directly to the given sparse
    matrix descriptor in csr format.

    If the `async` keyword is present, the op is executed asynchronously (i.e.
    it does not block until the execution has finished on the device). In
    that case, it returns a `!gpu.async.token` in addition to the environment.

    Example:

    ```mlir
    %token = gpu.set_csr_pointers async [%dep] %positions, %coordinates, %values
          : memref<?xf32>, memref<?xindex>, memref<?xindex>
    ```
  }];

  let arguments = (ins Variadic<GPU_AsyncToken>:$asyncDependencies,
                       Arg<GPU_SparseSpMatHandle>:$spmat,
                       AnyMemRef:$positions,
                       AnyMemRef:$coordinates,
		       AnyMemRef:$values);
  let results = (outs Optional<GPU_AsyncToken>:$asyncToken);

  let assemblyFormat = [{
    custom<AsyncDependencies>(type($asyncToken), $asyncDependencies)
      $spmat `,` $positions `,` $coordinates `,` $values attr-dict
        `:` type($positions) `,` type($coordinates) `,` type($values)
  }];
}

def GPU_WarpExecuteOnLane0Op : GPU_Op<"warp_execute_on_lane_0",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface, ["areTypesCompatible"]>,
       SingleBlockImplicitTerminator<"gpu::YieldOp">,
       RecursiveMemoryEffects]> {
  let summary = "Executes operations in the associated region on thread #0 of a"
                "SPMD program";
  let description = [{
    `warp_execute_on_lane_0` is an operation used to bridge the gap between
    vector programming and SPMD programming model like GPU SIMT. It allows to
    trivially convert a region of vector code meant to run on a multiple threads
    into a valid SPMD region and then allows incremental transformation to
    distribute vector operations on the threads.

    Any code present in the region would only be executed on first thread/lane
    based on the `laneid` operand. The `laneid` operand is an integer ID between
    [0, `warp_size`). The `warp_size` attribute indicates the number of lanes in
    a warp.

    Operands are vector values distributed on all lanes that may be used by
    the single lane execution. The matching region argument is a vector of all
    the values of those lanes available to the single active lane. The
    distributed dimension is implicit based on the shape of the operand and
    argument. the properties of the distribution may be described by extra
    attributes (e.g. affine map).

    Return values are distributed on all lanes using laneId as index. The
    vector is distributed based on the shape ratio between the vector type of
    the yield and the result type.
    If the shapes are the same this means the value is broadcasted to all lanes.
    In the future the distribution can be made more explicit using affine_maps
    and will support having multiple Ids.

    Therefore the `warp_execute_on_lane_0` operations allow to implicitly copy
    between lane0 and the lanes of the warp. When distributing a vector
    from lane0 to all the lanes, the data are distributed in a block cyclic way.
    For example `vector<64xf32>` gets distributed on 32 threads and map to
    `vector<2xf32>` where thread 0 contains vector[0] and vector[1].

    During lowering values passed as operands and return value need to be
    visible to different lanes within the warp. This would usually be done by
    going through memory.

    The region is *not* isolated from above. For values coming from the parent
    region not going through operands only the lane 0 value will be accesible so
    it generally only make sense for uniform values.

    Example:
    ```
    // Execute in parallel on all threads/lanes.
    gpu.warp_execute_on_lane_0 (%laneid)[32] {
      // Serial code running only on thread/lane 0.
      ...
    }
    // Execute in parallel on all threads/lanes.
    ```

    This may be lowered to an scf.if region as below:
    ```
      // Execute in parallel on all threads/lanes.
      %cnd = arith.cmpi eq, %laneid, %c0 : index
      scf.if %cnd {
        // Serial code running only on thread/lane 0.
        ...
      }
      // Execute in parallel on all threads/lanes.
    ```

    When the region has operands and/or return values:
    ```
    // Execute in parallel on all threads/lanes.
    %0 = gpu.warp_execute_on_lane_0(%laneid)[32]
    args(%v0 : vector<4xi32>) -> (vector<1xf32>) {
    ^bb0(%arg0 : vector<128xi32>) :
      // Serial code running only on thread/lane 0.
      ...
      gpu.yield %1 : vector<32xf32>
    }
    // Execute in parallel on all threads/lanes.
    ```

    values at the region boundary would go through memory:
    ```
    // Execute in parallel on all threads/lanes.
    ...
    // Store the data from each thread into memory and Synchronization.
    %tmp0 = memreg.alloc() : memref<128xf32>
    %tmp1 = memreg.alloc() : memref<32xf32>
    %cnd = arith.cmpi eq, %laneid, %c0 : index
    vector.store %v0, %tmp0[%laneid] : memref<128xf32>, vector<4xf32>
    some_synchronization_primitive
    scf.if %cnd {
      // Serialized code running only on thread 0.
      // Load the data from all the threads into a register from thread 0. This
      // allow threads 0 to access data from all the threads.
      %arg0 = vector.load %tmp0[%c0] : memref<128xf32>, vector<128xf32>
      ...
      // Store the data from thread 0 into memory.
      vector.store %1, %tmp1[%c0] : memref<32xf32>, vector<32xf32>
    }
    // Synchronization and load the data in a block cyclic way so that the
    // vector is distributed on all threads.
    some_synchronization_primitive
    %0 = vector.load %tmp1[%laneid] : memref<32xf32>, vector<32xf32>
    // Execute in parallel on all threads/lanes.
    ```

  }];

  let hasVerifier = 1;
  let hasCustomAssemblyFormat = 1;
  let arguments = (ins Index:$laneid, I64Attr:$warp_size,
                       Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$warpRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$laneid,
                   "int64_t":$warpSize)>,
    // `blockArgTypes` are different than `args` types as they are they
    // represent all the `args` instances visibile to lane 0. Therefore we need
    // to explicit pass the type.
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$laneid,
                   "int64_t":$warpSize, "ValueRange":$args,
                   "TypeRange":$blockArgTypes)>
  ];

  let extraClassDeclaration = [{
    bool isDefinedOutsideOfRegion(Value value) {
      return !getRegion().isAncestor(value.getParentRegion());
    }
  }];
}

#endif // GPU_OPS
