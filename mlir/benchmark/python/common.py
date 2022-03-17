"""Common utilities that are useful for all the benchmarks."""
import numpy as np

import mlir.all_passes_registration

from mlir import ir
from mlir.dialects import arith
from mlir.dialects import builtin
from mlir.dialects import func
from mlir.dialects import memref
from mlir.dialects import scf
from mlir.passmanager import PassManager


def setup_passes(mlir_module):
    """Setup pass pipeline parameters for benchmark functions.
    """
    opt = (
        "parallelization-strategy=0"
        " vectorization-strategy=0 vl=1 enable-simd-index32=False"
    )
    pipeline = f"sparse-compiler{{{opt}}}"
    PassManager.parse(pipeline).run(mlir_module)


def create_sparse_np_tensor(dimensions, number_of_elements):
    """Constructs a numpy tensor of dimensions `dimensions` that has only a
    specific number of nonzero elements, specified by the `number_of_elements`
    argument.
    """
    tensor = np.zeros(dimensions, np.float64)
    tensor_indices_list = [
        [np.random.randint(0, dimension) for dimension in dimensions]
        for _ in range(number_of_elements)
    ]
    for tensor_indices in tensor_indices_list:
        current_tensor = tensor
        for tensor_index in tensor_indices[:-1]:
            current_tensor = current_tensor[tensor_index]
        current_tensor[tensor_indices[-1]] = np.random.uniform(1, 100)
    return tensor


def get_kernel_func_from_module(module: ir.Module) -> func.FuncOp:
    """Takes an mlir module object and extracts the function object out of it.
    This function only works for a module with one region, one block, and one
    operation.
    """
    assert len(module.operation.regions) == 1, \
        "Expected kernel module to have only one region"
    assert len(module.operation.regions[0].blocks) == 1, \
        "Expected kernel module to have only one block"
    assert len(module.operation.regions[0].blocks[0].operations) == 1, \
        "Expected kernel module to have only one operation"
    return module.operation.regions[0].blocks[0].operations[0]


def emit_timer_func() -> func.FuncOp:
    """Returns the declaration of nano_time function. If nano_time function is
    used, the `MLIR_RUNNER_UTILS` and `MLIR_C_RUNNER_UTILS` must be included.
    """
    i64_type = ir.IntegerType.get_signless(64)
    nano_time = func.FuncOp(
        "nano_time", ([], [i64_type]), visibility="private")
    nano_time.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
    return nano_time


def emit_benchmark_wrapped_main_func(func, timer_func):
    """Takes a function and a timer function, both represented as FuncOp
    objects, and returns a new function. This new function wraps the call to
    the original function between calls to the timer_func and this wrapping
    in turn is executed inside a loop. The loop is executed
    len(func.type.results) times. This function can be used to create a
    "time measuring" variant of a function.
    """
    i64_type = ir.IntegerType.get_signless(64)
    memref_of_i64_type = ir.MemRefType.get([-1], i64_type)
    wrapped_func = func.FuncOp(
        # Same signature and an extra buffer of indices to save timings.
        "main",
        (func.arguments.types + [memref_of_i64_type], func.type.results),
        visibility="public"
    )
    wrapped_func.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

    num_results = len(func.type.results)
    with ir.InsertionPoint(wrapped_func.add_entry_block()):
        timer_buffer = wrapped_func.arguments[-1]
        zero = arith.ConstantOp.create_index(0)
        n_iterations = memref.DimOp(ir.IndexType.get(), timer_buffer, zero)
        one = arith.ConstantOp.create_index(1)
        iter_args = list(wrapped_func.arguments[-num_results - 1:-1])
        loop = scf.ForOp(zero, n_iterations, one, iter_args)
        with ir.InsertionPoint(loop.body):
            start = func.CallOp(timer_func, [])
            call = func.CallOp(
                func,
                wrapped_func.arguments[:-num_results - 1] + loop.inner_iter_args
            )
            end = func.CallOp(timer_func, [])
            time_taken = arith.SubIOp(end, start)
            memref.StoreOp(time_taken, timer_buffer, [loop.induction_variable])
            scf.YieldOp(list(call.results))
        func.ReturnOp(loop)

    return wrapped_func
