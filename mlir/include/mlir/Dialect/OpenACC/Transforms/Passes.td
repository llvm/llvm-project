//===-- Passes.td - OpenACC pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES
#define MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Common options shared by multiple ACC passes
//===----------------------------------------------------------------------===//

def AccDeviceTypeOption : Option<"deviceType", "device-type",
    "mlir::acc::DeviceType", "mlir::acc::DeviceType::None",
    "Target device type. One use case is ensuring that device_type-specific "
    "clauses are considered. Another is device-specific specializations.",
    [{::llvm::cl::values(
       clEnumValN(mlir::acc::DeviceType::None, "none", "none"),
       clEnumValN(mlir::acc::DeviceType::Host, "host", "host"),
       clEnumValN(mlir::acc::DeviceType::Multicore, "multicore", "multicore"),
       clEnumValN(mlir::acc::DeviceType::Nvidia, "nvidia", "nvidia"),
       clEnumValN(mlir::acc::DeviceType::Radeon, "radeon", "radeon"))
    }]>;

//===----------------------------------------------------------------------===//
// Pass definitions
//===----------------------------------------------------------------------===//

def LegalizeDataValuesInRegion : Pass<"openacc-legalize-data-values", "mlir::func::FuncOp"> {
  let summary = "Legalizes SSA values in compute regions with results from data clause operations";
  let description = [{
    This pass replace uses of the `varPtr` in compute regions (kernels,
    parallel, serial) with the result of data clause operations (`accPtr`).
  }];
  let options = [
    Option<"hostToDevice", "host-to-device", "bool", "true",
           "Replace varPtr uses with accPtr if true. Replace accPtr uses with "
           "varPtr if false">,
    Option<"applyToAccDataConstruct", "apply-to-acc-data-construct", "bool", "true",
           "Replaces varPtr uses with accPtr for acc compute regions contained "
           "within acc.data or acc.declare region.">
  ];
}

def ACCImplicitData : Pass<"acc-implicit-data", "mlir::ModuleOp"> {
  let summary = "Generate implicit data attributes for OpenACC compute constructs";
  let description = [{
    This pass implements the OpenACC specification for "Variables with
    Implicitly Determined Data Attributes" (OpenACC 3.4 spec, section 2.6.2).

    The pass automatically generates data clause operations for variables used
    within OpenACC compute constructs (parallel, kernels, serial) that do not
    already have explicit data clauses. The semantics follow these rules:

    1. If there is a default(none) clause visible, no implicit data actions
       apply.

    2. An aggregate variable (arrays, derived types, etc.) will be treated as:
       - In a present clause when default(present) is visible.
       - In a copy clause otherwise.

    3. A scalar variable will be treated as if it appears in:
       - A copy clause if the compute construct is a kernels construct.
       - A firstprivate clause otherwise (parallel, serial).
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect",
      "mlir::memref::MemRefDialect",
      "mlir::arith::ArithDialect"];
  let options = [
    Option<"enableImplicitReductionCopy", "enable-implicit-reduction-copy",
           "bool", "true",
           "Enable applying implicit copy in lieu of implicit firstprivate for "
           "reduction variables. This allows uniform treatment of reduction "
           "variables between combined constructs (e.g., 'parallel loop') and "
           "separate constructs (e.g., 'parallel' followed by 'loop'), where "
           "the OpenACC spec requires copy semantics for the former but "
           "firstprivate would normally apply for the latter.">
  ];
}

def ACCImplicitDeclare : Pass<"acc-implicit-declare", "mlir::ModuleOp"> {
  let summary = "Applies implicit acc declare to globals referenced in compute and routine acc regions";
  let description = [{
    This pass applies implicit `acc declare` actions to global variables
    referenced in OpenACC compute regions and routine functions.

    The pass performs the following actions:

    1. Hoists address-of operations for non-constant globals out of OpenACC
       regions when they can be implicitly mapped rather than declared.

    2. Collects global symbols referenced in:
       - OpenACC compute constructs (parallel, kernels, serial)
       - Functions marked with acc routine
       - Initialization regions of existing acc declare globals
       - Private/firstprivate/reduction recipe operations

    3. Marks collected globals with the acc.declare attribute using the
       copyin data clause.

    The pass avoids unnecessary declare marking by:
    - Skipping function symbols (which use acc routine instead)
    - Hoisting non-constant global references that can use implicit mapping
    - Only processing symbols that are not already valid in device regions
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect"];
}

def ACCImplicitRoutine : Pass<"acc-implicit-routine", "mlir::ModuleOp"> {
  let summary = "Generate implicit acc routine for functions in acc regions";
  let description = [{
    This pass implements the implicit rules described in OpenACC specification
    for `Routine Directive` (OpenACC 3.4 spec, section 2.15.1).

    "If no explicit routine directive applies to a procedure whose definition
    appears in the program unit being compiled, then the implementation applies
    an implicit routine directive to that procedure if any of the following
    conditions holds:
    - The procedure is called or its address is accessed in a compute region."

    The specification further states:
    "When the implementation applies an implicit routine directive to a procedure,
    it must recursively apply implicit routine directives to other procedures for
    which the above rules specify relevant dependencies. Such dependencies can
    form a cycle, so the implementation must take care to avoid infinite recursion."

    This pass implements these requirements by:
    1. Walking through all OpenACC compute constructs and functions already
       marked with `acc routine` in the module and identifying function calls
       within these regions.
    2. Creating implicit `acc.routine` operations for functions that don't already
       have routine declarations.
    3. Recursively walking through all existing `acc routine` and creating
       implicit routine operations for function calls within these routines,
       while avoiding infinite recursion through proper tracking.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect"];
  let options = [ AccDeviceTypeOption ];
}

def ACCDeclareGPUModuleInsertion : Pass<"acc-declare-gpu-module-insertion", "mlir::ModuleOp"> {
  let summary = "Copy globals with acc.declare into the GPU module";
  let description = [{
    Copies globals that have the `acc.declare` attribute into the GPU module so
    that device code can reference them.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect", "mlir::gpu::GPUDialect"];
}

def ACCLegalizeSerial : Pass<"acc-legalize-serial", "mlir::func::FuncOp"> {
  let summary = "Legalize OpenACC serial constructs";
  let description = [{
    This pass converts `acc.serial` constructs into `acc.parallel` constructs
    with `num_gangs(1)`, `num_workers(1)`, and `vector_length(1)`.

    This transformation simplifies processing of acc regions by unifying the
    handling of serial and parallel constructs. Since an OpenACC serial region
    executes sequentially (like a parallel region with a single gang, worker,
    and vector), this conversion is semantically equivalent while enabling code
    reuse in later compilation stages.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect",
      "mlir::arith::ArithDialect"];
}


def ACCLoopTiling : Pass<"acc-loop-tiling", "mlir::func::FuncOp"> {
  let summary = "Tile OpenACC loops with tile clauses";
  let description = [{
    This pass implements loop tiling transformations for OpenACC loops that
    have tile clauses. The pass transforms loops with `tile(size1, size2, ...)`
    clauses into tiled loop nests.

    For a 2-level nested loop with tile(T1, T2), the transformation produces:
    - Outer tile loops that iterate over tiles
    - Inner element loops that iterate within each tile

    Example transformation:
    ```
    // Before:
    #pragma acc loop tile(32, 32)
    for (i = 0; i < N; i++)
      for (j = 0; j < M; j++)
        A[i][j] = ...

    // After:
    for (i = 0; i < N; i += 32)           // tile loop 1
      for (j = 0; j < M; j += 32)         // tile loop 2
        for (ii = i; ii < min(N, i+32); ii++)    // element loop 1
          for (jj = j; jj < min(M, j+32); jj++)  // element loop 2
            A[ii][jj] = ...
    ```

    The pass handles:
    - Constant tile sizes
    - Wildcard tile sizes ('*') which use a default tile size
    - Collapsed loops with tile counts exceeding collapse count
    - Proper handling of loop attributes (gang, worker, vector)
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect",
      "mlir::arith::ArithDialect"];
  let options = [
    Option<"defaultTileSize", "default-tile-size", "int32_t", "32",
           "Default tile size to use for wildcard ('*') tile sizes">
  ];
}

def ACCSpecializeForDevice : Pass<"acc-specialize-for-device", "mlir::func::FuncOp"> {
  let summary = "Strip OpenACC constructs inside device code";
  let description = [{
    In a specialized acc routine or compute construct, many OpenACC operations
    do not make sense because they are host-side constructs. This pass removes
    or transforms these operations appropriately.

    The following operations are handled:
    - Data entry ops (replaced with var): acc.attach, acc.copyin, acc.create,
      acc.declare_device_resident, acc.declare_link, acc.deviceptr,
      acc.get_deviceptr, acc.nocreate, acc.present, acc.update_device,
      acc.use_device
    - Data exit ops (erased): acc.copyout, acc.delete, acc.detach,
      acc.update_host
    - Structured data (inline region): acc.data, acc.host_data,
      acc.kernel_environment
    - Unstructured data (erased): acc.enter_data, acc.exit_data, acc.update,
      acc.declare_enter, acc.declare_exit
    - Compute constructs (inline region): acc.parallel, acc.serial, acc.kernels
    - Runtime ops (erased): acc.init, acc.shutdown, acc.set, acc.wait
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect"];
}

def ACCSpecializeForHost : Pass<"acc-specialize-for-host", "mlir::func::FuncOp"> {
  let summary = "Convert OpenACC operations for host execution";
  let description = [{
    This pass converts OpenACC operations to host-compatible representations.
    It serves as a conversion pass that transforms ACC constructs to enable
    execution on the host rather than on accelerator devices.

    There are two modes of operation:

    1. Default mode (orphan operations only): Only orphan operations that are
       not allowed outside compute regions are converted. Structured/unstructured
       data constructs, compute constructs, and their associated data operations
       are NOT removed.

    2. Host fallback mode (enableHostFallback=true): ALL ACC operations within
       the region are converted to host equivalents. This is used when the `if`
       clause evaluates to false at runtime.

    The following operations are handled:
    - Atomic ops: converted to load/store operations
    - Loop ops: converted to scf.for or scf.execute_region
    - Data entry ops (orphan): replaced with var operand
    - In host fallback mode: all data, compute, and runtime ops are removed
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect",
      "mlir::scf::SCFDialect"];
  let options = [
    Option<"enableHostFallback", "enable-host-fallback", "bool", "false",
           "Enable host fallback mode which converts ALL ACC operations, "
           "not just orphan operations. Use this when the `if` clause "
           "evaluates to false.">
  ];
}

def ACCIfClauseLowering : Pass<"acc-if-clause-lowering", "mlir::func::FuncOp"> {
  let summary = "Lower if clauses in ACC compute constructs";
  let description = [{
    This pass lowers OpenACC compute constructs (parallel, kernels, serial) with
    `if` clauses using region specialization. It creates two execution paths:
    device execution when the condition is true, host execution when false.

    When an ACC compute construct has an `if` clause, the construct should only
    execute on the device when the condition is true. If the condition is false,
    the code should execute on the host instead. This pass transforms:

    ```mlir
    acc.parallel if(%cond) { ... }
    ```

    Into:

    ```mlir
    scf.if %cond {
      // Device path: clone data ops, compute construct without if, exit ops
      acc.parallel { ... }
    } else {
      // Host path: original region body with ACC ops converted to host
    }
    ```

    The transformation handles:
    - Data entry operations (acc.copyin, acc.create, etc.) are cloned to device path
    - Data exit operations (acc.copyout, acc.delete, etc.) are cloned to device path
    - The host path uses `populateACCHostFallbackPatterns` to convert ACC ops
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect",
      "mlir::scf::SCFDialect"];
}

def OffloadLiveInValueCanonicalization : Pass<"offload-livein-value-canonicalization", "mlir::func::FuncOp"> {
  let summary = "Canonicalize live-in values for regions destined for outlining";
  let description = [{
    This pass canonicalizes live-in values for regions destined for outlining.
    It handles operations that produce synthetic types or values that cannot
    be passed as arguments to outlined regions.

    The pass performs the following transformations:

    1. **Sinking**: Operations whose results are only used inside the region
       are moved into the region. This reduces the number of live-in values
       and keeps related operations together.

    2. **Rematerialization**: Operations whose results are used both inside
       and outside the region are cloned into the region. The uses inside
       the region are updated to use the cloned operation's results.

    Operations are considered candidates for these transformations if they
    implement the `OutlineRematerializationOpInterface` or match constant
    patterns. These operations typically produce synthetic types (shapes,
    bounds, field indices) that cannot be passed as function arguments.

    The pass iterates until convergence since canonicalizing one value may
    expose new candidates (e.g., a bounds operation's operands may themselves
    be constants that should be rematerialized).

    Example transformation (rematerialization):
    ```mlir
    // Before:
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c10 : index)
    acc.parallel {
      %priv = acc.private varPtr(%ptr : ...) bounds(%bounds) -> ...
      acc.yield
    }
    // %bounds is also used elsewhere

    // After:
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c10 : index)
    acc.parallel {
      %c0_clone = arith.constant 0 : index
      %c10_clone = arith.constant 10 : index
      %bounds_clone = acc.bounds lowerbound(%c0_clone : index) upperbound(%c10_clone : index)
      %priv = acc.private varPtr(%ptr : ...) bounds(%bounds_clone) -> ...
      acc.yield
    }
    ```

    Example transformation (sinking):
    ```mlir
    // Before:
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c10 : index)
    acc.parallel {
      %priv = acc.private varPtr(%ptr : ...) bounds(%bounds) -> ...
      acc.yield
    }
    // %bounds is NOT used elsewhere

    // After:
    acc.parallel {
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %bounds = acc.bounds lowerbound(%c0 : index) upperbound(%c10 : index)
      %priv = acc.private varPtr(%ptr : ...) bounds(%bounds) -> ...
      acc.yield
    }
    ```
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect"];
}

def ACCRecipeMaterialization : Pass<"acc-recipe-materialization", "mlir::ModuleOp"> {
  let summary = "Materialize OpenACC private, firstprivate and reduction recipes";
  let description = [{
    Materializes OpenACC privatization, firstprivate and reduction recipes by
    cloning init, copy, combiner, and destroy into the construct. Replaces recipe
    references with materialized values (including acc.reduction_init and
    acc.reduction_combine_region for reductions) and removes unused recipe
    symbols.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect", "mlir::arith::ArithDialect"];
}

def OffloadTargetVerifier : Pass<"offload-target-verifier", "mlir::func::FuncOp"> {
  let summary = "Verify values and symbols live into offload regions for legality";
  let description = [{
    This pass verifies that values and symbols used within OpenACC compute
    constructs and other offload regions are legal for the target execution
    model.

    The pass performs two main checks:

    1. **Live-in Value Verification**: Checks that all values that are live
       into an offload region are valid for use in that region. This includes
       checking that pointer-like and mappable types have appropriate data
       clauses or device attributes.

    2. **Symbol Use Verification**: Checks that all symbol references within
       an offload region are valid for that region. This includes checking for
       acc.routine declarations and acc.declare attributes.

    The device_type option notes the target execution model:
    - `none`, `nvidia`, `radeon`: Device execution (GPU offload)
    - `host`, `multicore`: Host execution

    When soft_check is enabled, the pass only emits debug messages for illegal
    values/symbols instead of failing compilation. This is useful for
    diagnostic purposes.
  }];
  let dependentDialects = ["mlir::acc::OpenACCDialect"];
  let options = [
    AccDeviceTypeOption,
    Option<"softCheck", "soft-check", "bool", "false",
           "When true, illegal values are printed via LLVM_DEBUG instead of "
           "failing compilation. Useful for diagnostic purposes.">
  ];
}

def ACCComputeLowering : Pass<"acc-compute-lowering", "mlir::func::FuncOp"> {
  let summary = "Lower ACC compute constructs to acc.compute_region";
  let description = [{
    Converts ACC frontend compute constructs (`acc.parallel`, `acc.kernels`,
    `acc.serial`) to `acc.compute_region` wrapped in `acc.kernel_environment`.
    Converts `acc.loop` to SCF parallel/for loops with parallel dimension
    annotations.

    The pass applies two phases of pattern rewrites:
    1. Loop conversion: `acc.loop` is converted to `scf.parallel` or `scf.for`
       while the parent compute construct is still present (needed to determine
       loop conversion strategy).
    2. Compute construct conversion: `acc.parallel`, `acc.kernels`, and
       `acc.serial` are replaced by `acc.kernel_environment` containing
       `acc.compute_region`.
  }];
  let dependentDialects = [
    "mlir::acc::OpenACCDialect",
    "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [ AccDeviceTypeOption ];
}

def ACCRoutineLowering : Pass<"acc-routine-lowering", "mlir::ModuleOp"> {
  let summary = "Specialize `acc routine` functions for device";
  let description = [{
    This pass handles `acc routine` directive by creating specialized
    functions with appropriate parallelism information that can be used for
    eventual creation of device function.

    For each acc.routine that is not bound by name, the pass creates a new
    function (the "device" copy) whose body is a single acc.compute_region
    containing a clone of the original (host) function body. Parallelism is
    expressed by one acc.par_width derived from the routine's clauses (seq,
    vector, worker, gang). The pass does not use acc.kernel_environment. It
    sets acc.specialized_routine on the new function and updates the
    acc.routine's func_name to point to it. For nohost routines, all uses of
    the host symbol are replaced with the device symbol and the host function
    is erased. Routines with bind(name) and external functions are skipped.
  }];
  let dependentDialects = [
    "mlir::acc::OpenACCDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect"
  ];
  let options = [ AccDeviceTypeOption ];
}

def ACCRoutineToGPUFunc : Pass<"acc-routine-to-gpu-func", "mlir::ModuleOp"> {
  let summary = "Move ACC routine functions into the GPU module as gpu.func";
  let description = [{
    This pass moves functions associated with `acc routine` (and any callees
    that must be present on the device) into the GPU module as `gpu.func`
    operations.
  }];
  let dependentDialects = ["mlir::gpu::GPUDialect"];
  let options = [ AccDeviceTypeOption ];
}

#endif // MLIR_DIALECT_OPENACC_TRANSFORMS_PASSES
