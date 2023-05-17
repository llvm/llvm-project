"""This file contains benchmarks for sparse tensors. In particular, it
contains benchmarks for both mlir sparse tensor dialect and numpy so that they
can be compared against each other.
"""
import ctypes
import numpy as np
import os
import re
import time

from mlir import ir
from mlir import runtime as rt
from mlir.dialects import func
from mlir.dialects.linalg.opdsl import lang as dsl
from mlir.execution_engine import ExecutionEngine

from common import create_sparse_np_tensor
from common import emit_timer_func
from common import emit_benchmark_wrapped_main_func
from common import get_kernel_func_from_module
from common import setup_passes


@dsl.linalg_structured_op
def matmul_dsl(
    A=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.K),
    B=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.N),
    C=dsl.TensorDef(dsl.T, dsl.S.M, dsl.S.N, output=True)
):
    """Helper function for mlir sparse matrix multiplication benchmark."""
    C[dsl.D.m, dsl.D.n] += A[dsl.D.m, dsl.D.k] * B[dsl.D.k, dsl.D.n]


def benchmark_sparse_mlir_multiplication():
    """Benchmark for mlir sparse matrix multiplication. Because its an
    MLIR benchmark we need to return both a `compiler` function and a `runner`
    function.
    """
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        f64 = ir.F64Type.get()
        param1_type = ir.RankedTensorType.get([1000, 1500], f64)
        param2_type = ir.RankedTensorType.get([1500, 2000], f64)
        result_type = ir.RankedTensorType.get([1000, 2000], f64)
        with ir.InsertionPoint(module.body):
            @func.FuncOp.from_py_func(param1_type, param2_type, result_type)
            def sparse_kernel(x, y, z):
                return matmul_dsl(x, y, outs=[z])

    def compiler():
        with ir.Context(), ir.Location.unknown():
            kernel_func = get_kernel_func_from_module(module)
            timer_func = emit_timer_func()
            wrapped_func = emit_benchmark_wrapped_main_func(
                kernel_func,
                timer_func
            )
            main_module_with_benchmark = ir.Module.parse(
                str(timer_func) + str(wrapped_func) + str(kernel_func)
            )
            setup_passes(main_module_with_benchmark)
            c_runner_utils = os.getenv("MLIR_C_RUNNER_UTILS", "")
            assert os.path.exists(c_runner_utils),\
                f"{c_runner_utils} does not exist." \
                f" Please pass a valid value for" \
                f" MLIR_C_RUNNER_UTILS environment variable."
            runner_utils = os.getenv("MLIR_RUNNER_UTILS", "")
            assert os.path.exists(runner_utils),\
                f"{runner_utils} does not exist." \
                f" Please pass a valid value for MLIR_RUNNER_UTILS" \
                f" environment variable."

            engine = ExecutionEngine(
                main_module_with_benchmark,
                3,
                shared_libs=[c_runner_utils, runner_utils]
            )
            return engine.invoke

    def runner(engine_invoke):
        compiled_program_args = []
        for argument_type in [
            result_type, param1_type, param2_type, result_type
        ]:
            argument_type_str = str(argument_type)
            dimensions_str = re.sub("<|>|tensor", "", argument_type_str)
            dimensions = [int(dim) for dim in dimensions_str.split("x")[:-1]]
            if argument_type == result_type:
                argument = np.zeros(dimensions, np.float64)
            else:
                argument = create_sparse_np_tensor(dimensions, 1000)
            compiled_program_args.append(
                ctypes.pointer(
                    ctypes.pointer(rt.get_ranked_memref_descriptor(argument))
                )
            )
        np_timers_ns = np.array([0], dtype=np.int64)
        compiled_program_args.append(
            ctypes.pointer(
                ctypes.pointer(rt.get_ranked_memref_descriptor(np_timers_ns))
            )
        )
        engine_invoke("main", *compiled_program_args)
        return int(np_timers_ns[0])

    return compiler, runner


def benchmark_np_matrix_multiplication():
    """Benchmark for numpy matrix multiplication. Because its a python
    benchmark, we don't have any `compiler` function returned. We just return
    the `runner` function.
    """
    def runner():
        argument1 = np.random.uniform(low=0.0, high=100.0, size=(1000, 1500))
        argument2 = np.random.uniform(low=0.0, high=100.0, size=(1500, 2000))
        start_time = time.time_ns()
        np.matmul(argument1, argument2)
        return time.time_ns() - start_time

    return None, runner
