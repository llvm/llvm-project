# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import x86vector


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
def non_configurable_patterns():
    # CHECK-LABEL: TEST: non_configurable_patterns
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.x86vector.vector_contract_to_fma
    x86vector.ApplyVectorContractToFMAPatternsOp()
    # CHECK: transform.apply_patterns.x86vector.vector_contract_to_packed_type_dot_product
    x86vector.ApplyVectorContractToPackedTypeDotProductPatternsOp()
    # CHECK: transform.apply_patterns.x86vector.vector_contract_bf16_to_fma
    x86vector.ApplyVectorContractBF16ToFMAPatternsOp()
    # CHECK: transform.apply_patterns.x86vector.sink_vector_producer_ops
    x86vector.ApplySinkVectorProducerOpsPatternsOp()
    # CHECK: transform.apply_patterns.x86vector.shuffle_vector_fma_ops
    x86vector.ApplyShuffleVectorFMAOpsPatternsOp()
