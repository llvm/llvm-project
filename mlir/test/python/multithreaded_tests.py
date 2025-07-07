# RUN: %PYTHON %s
"""
This script generates multi-threaded tests to check free-threading mode using CPython compiled with TSAN.
Tests can be run using pytest:
```bash
python3.13t -mpytest -vvv multithreaded_tests.py
```

IMPORTANT. Running tests are not checking the correctness, but just the execution of the tests in multi-threaded context
and passing if no warnings reported by TSAN and failing otherwise.


Details on the generated tests and execution:
1) Multi-threaded execution: all generated tests are executed independently by
a pool of threads, running each test multiple times, see @multi_threaded for details

2) Tests generation: we use existing tests: test/python/ir/*.py,
test/python/dialects/*.py, etc to generate multi-threaded tests.
In details, we perform the following:
a) we define a list of source tests to be used to generate multi-threaded tests, see `TEST_MODULES`.
b) we define `TestAllMultiThreaded` class and add existing tests to the class. See `add_existing_tests` method.
c) for each test file, we copy and modify it: test/python/ir/affine_expr.py -> /tmp/ir/affine_expr.py.
In order to import the test file as python module, we remove all executing functions, like
`@run` or `run(testMethod)`. See `copy_and_update` and `add_existing_tests` methods for details.


Observed warnings reported by TSAN.

CPython and free-threading known data-races:
1) ctypes related races: https://github.com/python/cpython/issues/127945
2) LLVM related data-races, llvm::raw_ostream is not thread-safe
- mlir pass manager
- dialects/transform_interpreter.py
- ir/diagnostic_handler.py
- ir/module.py
3) Dialect gpu module-to-binary method is unsafe
"""
import concurrent.futures
import gc
import importlib.util
import os
import sys
import threading
import tempfile
import unittest

from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Optional, List

import mlir.dialects.arith as arith
from mlir.dialects import transform
from mlir.ir import Context, Location, Module, IntegerType, InsertionPoint


def import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def copy_and_update(src_filepath: Path, dst_filepath: Path):
    # We should remove all calls like `run(testMethod)`
    with open(src_filepath, "r") as reader, open(dst_filepath, "w") as writer:
        while True:
            src_line = reader.readline()
            if len(src_line) == 0:
                break
            skip_lines = [
                "run(",
                "@run",
                "@constructAndPrintInModule",
                "run_apply_patterns(",
                "@run_apply_patterns",
                "@test_in_context",
                "@construct_and_print_in_module",
            ]
            if any(src_line.startswith(line) for line in skip_lines):
                continue
            writer.write(src_line)


# Helper run functions
def run(f):
    f()


def run_with_context_and_location(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        f()
    return f


def run_with_insertion_point(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f(ctx)
        print(module)


def run_with_insertion_point_v2(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            f()
        print(module)
    return f


def run_with_insertion_point_v3(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f(module)
        print(module)
    return f


def run_with_insertion_point_v4(f):
    print("\nTEST:", f.__name__)
    with Context() as ctx, Location.unknown():
        ctx.allow_unregistered_dialects = True
        module = Module.create()
        with InsertionPoint(module.body):
            f()
    return f


def run_apply_patterns(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                apply = transform.ApplyPatternsOp(sequence.bodyTarget)
                with InsertionPoint(apply.patterns):
                    f()
                transform.YieldOp()
        print("\nTEST:", f.__name__)
        print(module)
    return f


def run_transform_tensor_ext(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            sequence = transform.SequenceOp(
                transform.FailurePropagationMode.Propagate,
                [],
                transform.AnyOpType.get(),
            )
            with InsertionPoint(sequence.body):
                f(sequence.bodyTarget)
                transform.YieldOp()
        print(module)
    return f


def run_transform_structured_ext(f):
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            print("\nTEST:", f.__name__)
            f()
        module.operation.verify()
        print(module)
    return f


def run_construct_and_print_in_module(f):
    print("\nTEST:", f.__name__)
    with Context(), Location.unknown():
        module = Module.create()
        with InsertionPoint(module.body):
            module = f(module)
        if module is not None:
            print(module)
    return f


TEST_MODULES = [
    ("execution_engine", run),
    ("pass_manager", run),
    ("dialects/affine", run_with_insertion_point_v2),
    ("dialects/func", run_with_insertion_point_v2),
    ("dialects/arith_dialect", run),
    ("dialects/arith_llvm", run),
    ("dialects/async_dialect", run),
    ("dialects/builtin", run),
    ("dialects/cf", run_with_insertion_point_v4),
    ("dialects/complex_dialect", run),
    ("dialects/func", run_with_insertion_point_v2),
    ("dialects/index_dialect", run_with_insertion_point),
    ("dialects/llvm", run_with_insertion_point_v2),
    ("dialects/math_dialect", run),
    ("dialects/memref", run),
    ("dialects/ml_program", run_with_insertion_point_v2),
    ("dialects/nvgpu", run_with_insertion_point_v2),
    ("dialects/nvvm", run_with_insertion_point_v2),
    ("dialects/ods_helpers", run),
    ("dialects/openmp_ops", run_with_insertion_point_v2),
    ("dialects/pdl_ops", run_with_insertion_point_v2),
    # ("dialects/python_test", run),  # TODO: Need to pass pybind11 or nanobind argv
    ("dialects/quant", run),
    ("dialects/rocdl", run_with_insertion_point_v2),
    ("dialects/scf", run_with_insertion_point_v2),
    ("dialects/shape", run),
    ("dialects/spirv_dialect", run),
    ("dialects/tensor", run),
    # ("dialects/tosa", ),  # Nothing to test
    ("dialects/transform_bufferization_ext", run_with_insertion_point_v2),
    # ("dialects/transform_extras", ),  # Needs a more complicated execution schema
    ("dialects/transform_gpu_ext", run_transform_tensor_ext),
    (
        "dialects/transform_interpreter",
        run_with_context_and_location,
        ["print_", "transform_options", "failed", "include"],
    ),
    (
        "dialects/transform_loop_ext",
        run_with_insertion_point_v2,
        ["loopOutline"],
    ),
    ("dialects/transform_memref_ext", run_with_insertion_point_v2),
    ("dialects/transform_nvgpu_ext", run_with_insertion_point_v2),
    ("dialects/transform_sparse_tensor_ext", run_transform_tensor_ext),
    ("dialects/transform_structured_ext", run_transform_structured_ext),
    ("dialects/transform_tensor_ext", run_transform_tensor_ext),
    (
        "dialects/transform_vector_ext",
        run_apply_patterns,
        ["configurable_patterns"],
    ),
    ("dialects/transform", run_with_insertion_point_v3),
    ("dialects/vector", run_with_context_and_location),
    ("dialects/gpu/dialect", run_with_context_and_location),
    ("dialects/gpu/module-to-binary-nvvm", run_with_context_and_location),
    ("dialects/gpu/module-to-binary-rocdl", run_with_context_and_location),
    ("dialects/linalg/ops", run),
    # TO ADD: No proper tests in this dialects/linalg/opsdsl/*
    # ("dialects/linalg/opsdsl/*", ...),
    ("dialects/sparse_tensor/dialect", run),
    ("dialects/sparse_tensor/passes", run),
    ("integration/dialects/pdl", run_construct_and_print_in_module),
    ("integration/dialects/transform", run_construct_and_print_in_module),
    ("integration/dialects/linalg/opsrun", run),
    ("ir/affine_expr", run),
    ("ir/affine_map", run),
    ("ir/array_attributes", run),
    ("ir/attributes", run),
    ("ir/blocks", run),
    ("ir/builtin_types", run),
    ("ir/context_managers", run),
    ("ir/debug", run),
    ("ir/diagnostic_handler", run),
    ("ir/dialects", run),
    ("ir/exception", run),
    ("ir/insertion_point", run),
    ("ir/integer_set", run),
    ("ir/location", run),
    ("ir/module", run),
    ("ir/operation", run),
    ("ir/symbol_table", run),
    ("ir/value", run),
]

TESTS_TO_SKIP = [
    "test_execution_engine__testNanoTime_multi_threaded",  # testNanoTime can't run in multiple threads, even with GIL
    "test_execution_engine__testSharedLibLoad_multi_threaded",  # testSharedLibLoad can't run in multiple threads, even with GIL
    "test_dialects_arith_dialect__testArithValue_multi_threaded",  # RuntimeError: Value caster is already registered: <class 'dialects/arith_dialect.testArithValue.<locals>.ArithValue'>, even with GIL
    "test_ir_dialects__testAppendPrefixSearchPath_multi_threaded",  # PyGlobals::setDialectSearchPrefixes is not thread-safe, even with GIL. Strange usage of static PyGlobals vs python exposed _cext.globals
    "test_ir_value__testValueCasters_multi_threaded",  # RuntimeError: Value caster is already registered: <function testValueCasters.<locals>.dont_cast_int, even with GIL
    # tests indirectly calling thread-unsafe llvm::raw_ostream
    "test_execution_engine__testInvalidModule_multi_threaded",  # mlirExecutionEngineCreate calls thread-unsafe llvm::raw_ostream
    "test_pass_manager__testPrintIrAfterAll_multi_threaded",  # IRPrinterInstrumentation::runAfterPass calls thread-unsafe llvm::raw_ostream
    "test_pass_manager__testPrintIrBeforeAndAfterAll_multi_threaded",  # IRPrinterInstrumentation::runBeforePass calls thread-unsafe llvm::raw_ostream
    "test_pass_manager__testPrintIrLargeLimitElements_multi_threaded",  # IRPrinterInstrumentation::runAfterPass calls thread-unsafe llvm::raw_ostream
    "test_pass_manager__testPrintIrTree_multi_threaded",  # IRPrinterInstrumentation::runAfterPass calls thread-unsafe llvm::raw_ostream
    "test_pass_manager__testRunPipeline_multi_threaded",  # PrintOpStatsPass::printSummary calls thread-unsafe llvm::raw_ostream
    "test_dialects_transform_interpreter__include_multi_threaded",  # mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) calls thread-unsafe llvm::raw_ostream
    "test_dialects_transform_interpreter__transform_options_multi_threaded",  # mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) calls thread-unsafe llvm::raw_ostream
    "test_dialects_transform_interpreter__print_self_multi_threaded",  # mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) call thread-unsafe llvm::raw_ostream
    "test_ir_diagnostic_handler__testDiagnosticCallbackException_multi_threaded",  # mlirEmitError calls thread-unsafe llvm::raw_ostream
    "test_ir_module__testParseSuccess_multi_threaded",  # mlirOperationDump calls thread-unsafe llvm::raw_ostream
    # False-positive TSAN detected race in llvm::RuntimeDyldELF::registerEHFrames()
    # Details: https://github.com/llvm/llvm-project/pull/107103/files#r1905726947
    "test_execution_engine__testCapsule_multi_threaded",
    "test_execution_engine__testDumpToObjectFile_multi_threaded",
]

TESTS_TO_XFAIL = [
    # execution_engine tests:
    # - ctypes related data-races: https://github.com/python/cpython/issues/127945
    "test_execution_engine__testBF16Memref_multi_threaded",
    "test_execution_engine__testBasicCallback_multi_threaded",
    "test_execution_engine__testComplexMemrefAdd_multi_threaded",
    "test_execution_engine__testComplexUnrankedMemrefAdd_multi_threaded",
    "test_execution_engine__testDynamicMemrefAdd2D_multi_threaded",
    "test_execution_engine__testF16MemrefAdd_multi_threaded",
    "test_execution_engine__testF8E5M2Memref_multi_threaded",
    "test_execution_engine__testInvokeFloatAdd_multi_threaded",
    "test_execution_engine__testInvokeVoid_multi_threaded",  # a ctypes race
    "test_execution_engine__testMemrefAdd_multi_threaded",
    "test_execution_engine__testRankedMemRefCallback_multi_threaded",
    "test_execution_engine__testRankedMemRefWithOffsetCallback_multi_threaded",
    "test_execution_engine__testUnrankedMemRefCallback_multi_threaded",
    "test_execution_engine__testUnrankedMemRefWithOffsetCallback_multi_threaded",
    # dialects tests
    "test_dialects_memref__testSubViewOpInferReturnTypeExtensiveSlicing_multi_threaded",  # Related to ctypes data races
    "test_dialects_transform_interpreter__print_other_multi_threaded",  # Fatal Python error: Aborted or mlir::transform::PrintOp::apply(mlir::transform::TransformRewriter...) is not thread-safe
    "test_dialects_gpu_module-to-binary-rocdl__testGPUToASMBin_multi_threaded",  # Due to global llvm-project/llvm/lib/Target/AMDGPU/GCNSchedStrategy.cpp::GCNTrackers variable mutation
    "test_dialects_gpu_module-to-binary-nvvm__testGPUToASMBin_multi_threaded",
    "test_dialects_gpu_module-to-binary-nvvm__testGPUToLLVMBin_multi_threaded",
    "test_dialects_gpu_module-to-binary-rocdl__testGPUToLLVMBin_multi_threaded",
    # integration tests
    "test_integration_dialects_linalg_opsrun__test_elemwise_builtin_multi_threaded",  # Related to ctypes data races
    "test_integration_dialects_linalg_opsrun__test_elemwise_generic_multi_threaded",  # Related to ctypes data races
    "test_integration_dialects_linalg_opsrun__test_fill_builtin_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_fill_generic_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_fill_rng_builtin_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_fill_rng_generic_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_max_pooling_builtin_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_max_pooling_generic_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_min_pooling_builtin_multi_threaded",  # ctypes
    "test_integration_dialects_linalg_opsrun__test_min_pooling_generic_multi_threaded",  # ctypes
]


def add_existing_tests(test_modules, test_prefix: str = "_original_test"):
    def decorator(test_cls):
        this_folder = Path(__file__).parent.absolute()
        test_cls.output_folder = tempfile.TemporaryDirectory()
        output_folder = Path(test_cls.output_folder.name)

        for test_mod_info in test_modules:
            assert isinstance(test_mod_info, tuple) and len(test_mod_info) in (2, 3)
            if len(test_mod_info) == 2:
                test_module_name, exec_fn = test_mod_info
                test_pattern = None
            else:
                test_module_name, exec_fn, test_pattern = test_mod_info

            src_filepath = this_folder / f"{test_module_name}.py"
            dst_filepath = (output_folder / f"{test_module_name}.py").absolute()
            if not dst_filepath.parent.exists():
                dst_filepath.parent.mkdir(parents=True)
            copy_and_update(src_filepath, dst_filepath)
            test_mod = import_from_path(test_module_name, dst_filepath)
            for attr_name in dir(test_mod):
                is_test_fn = test_pattern is None and attr_name.startswith("test")
                is_test_fn |= test_pattern is not None and any(
                    [p in attr_name for p in test_pattern]
                )
                if is_test_fn:
                    obj = getattr(test_mod, attr_name)
                    if callable(obj):
                        test_name = f"{test_prefix}_{test_module_name.replace('/', '_')}__{attr_name}"

                        def wrapped_test_fn(
                            self, *args, __test_fn__=obj, __exec_fn__=exec_fn, **kwargs
                        ):
                            __exec_fn__(__test_fn__)

                        setattr(test_cls, test_name, wrapped_test_fn)
        return test_cls

    return decorator


@contextmanager
def _capture_output(fp):
    # Inspired from jax test_utils.py capture_stderr method
    # ``None`` means nothing has not been captured yet.
    captured = None

    def get_output() -> str:
        if captured is None:
            raise ValueError("get_output() called while the context is active.")
        return captured

    with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as f:
        original_fd = os.dup(fp.fileno())
        os.dup2(f.fileno(), fp.fileno())
        try:
            yield get_output
        finally:
            # Python also has its own buffers, make sure everything is flushed.
            fp.flush()
            os.fsync(fp.fileno())
            f.seek(0)
            captured = f.read()
            os.dup2(original_fd, fp.fileno())


capture_stdout = partial(_capture_output, sys.stdout)
capture_stderr = partial(_capture_output, sys.stderr)


def multi_threaded(
    num_workers: int,
    num_runs: int = 5,
    skip_tests: Optional[List[str]] = None,
    xfail_tests: Optional[List[str]] = None,
    test_prefix: str = "_original_test",
    multithreaded_test_postfix: str = "_multi_threaded",
):
    """Decorator that runs a test in a multi-threaded environment."""

    def decorator(test_cls):
        for name, test_fn in test_cls.__dict__.copy().items():
            if not (name.startswith(test_prefix) and callable(test_fn)):
                continue

            name = f"test{name[len(test_prefix):]}"
            if skip_tests is not None:
                if any(
                    test_name.replace(multithreaded_test_postfix, "") in name
                    for test_name in skip_tests
                ):
                    continue

            def multi_threaded_test_fn(self, *args, __test_fn__=test_fn, **kwargs):
                with capture_stdout(), capture_stderr() as get_output:
                    barrier = threading.Barrier(num_workers)

                    def closure():
                        barrier.wait()
                        for _ in range(num_runs):
                            __test_fn__(self, *args, **kwargs)

                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=num_workers
                    ) as executor:
                        futures = []
                        for _ in range(num_workers):
                            futures.append(executor.submit(closure))
                        # We should call future.result() to re-raise an exception if test has
                        # failed
                        assert len(list(f.result() for f in futures)) == num_workers

                    gc.collect()
                    assert Context._get_live_count() == 0

                captured = get_output()
                if len(captured) > 0 and "ThreadSanitizer" in captured:
                    raise RuntimeError(
                        f"ThreadSanitizer reported warnings:\n{captured}"
                    )

            test_new_name = f"{name}{multithreaded_test_postfix}"
            if xfail_tests is not None and test_new_name in xfail_tests:
                multi_threaded_test_fn = unittest.expectedFailure(
                    multi_threaded_test_fn
                )

            setattr(test_cls, test_new_name, multi_threaded_test_fn)

        return test_cls

    return decorator


@multi_threaded(
    num_workers=10,
    num_runs=20,
    skip_tests=TESTS_TO_SKIP,
    xfail_tests=TESTS_TO_XFAIL,
)
@add_existing_tests(test_modules=TEST_MODULES, test_prefix="_original_test")
class TestAllMultiThreaded(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "output_folder"):
            cls.output_folder.cleanup()

    def _original_test_create_context(self):
        with Context() as ctx:
            print(ctx._get_live_count())
            print(ctx._get_live_module_count())
            print(ctx._get_live_operation_count())
            print(ctx._get_live_operation_objects())
            print(ctx._get_context_again() is ctx)
            print(ctx._clear_live_operations())

    def _original_test_create_module_with_consts(self):
        py_values = [123, 234, 345]
        with Context() as ctx:
            module = Module.create(loc=Location.file("foo.txt", 0, 0))

            dtype = IntegerType.get_signless(64)
            with InsertionPoint(module.body), Location.name("a"):
                arith.constant(dtype, py_values[0])

            with InsertionPoint(module.body), Location.name("b"):
                arith.constant(dtype, py_values[1])

            with InsertionPoint(module.body), Location.name("c"):
                arith.constant(dtype, py_values[2])


if __name__ == "__main__":
    # Do not run the tests on CPython with GIL
    if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled():
        unittest.main()
