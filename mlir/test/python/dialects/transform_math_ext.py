# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import math


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


@run_apply_patterns
def testF32Expansion():
    math.ApplyF32ExpansionPatternsOp()
    # CHECK-LABEL: TEST: testF32Expansion
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.math.f32_expansion


@run_apply_patterns
def testPolynomialApproximation():
    math.ApplyPolynomialApproximationPatternsOp()
    # CHECK-LABEL: TEST: testPolynomialApproximation
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.math.polynomial_approximation{{$}}


@run_apply_patterns
def testPolynomialApproximationAvx2():
    math.ApplyPolynomialApproximationPatternsOp(enable_avx2=True)
    # CHECK-LABEL: TEST: testPolynomialApproximationAvx2
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.math.polynomial_approximation enable_avx2
