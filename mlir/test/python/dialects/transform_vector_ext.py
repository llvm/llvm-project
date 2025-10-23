# RUN: %PYTHON %s | FileCheck %s

from mlir.ir import *
from mlir.dialects import transform
from mlir.dialects.transform import vector


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
    # CHECK: transform.apply_patterns.vector.cast_away_vector_leading_one_dim
    vector.ApplyCastAwayVectorLeadingOneDimPatternsOp()
    # CHECK: transform.apply_patterns.vector.rank_reducing_subview_patterns
    vector.ApplyRankReducingSubviewPatternsOp()
    # CHECK: transform.apply_patterns.vector.transfer_permutation_patterns
    vector.ApplyTransferPermutationPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_broadcast
    vector.ApplyLowerBroadcastPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_masks
    vector.ApplyLowerMasksPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_masked_transfers
    vector.ApplyLowerMaskedTransfersPatternsOp()
    # CHECK: transform.apply_patterns.vector.materialize_masks
    vector.ApplyMaterializeMasksPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_outerproduct
    vector.ApplyLowerOuterProductPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_gather
    vector.ApplyLowerGatherPatternsOp()
    # CHECK: transform.apply_patterns.vector.unroll_from_elements
    vector.ApplyUnrollFromElementsPatternsOp()
    # CHECK: transform.apply_patterns.vector.unroll_to_elements
    vector.ApplyUnrollToElementsPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_scan
    vector.ApplyLowerScanPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_shape_cast
    vector.ApplyLowerShapeCastPatternsOp()


@run_apply_patterns
def configurable_patterns():
    # CHECK-LABEL: TEST: configurable_patterns
    # CHECK: apply_patterns
    # CHECK: transform.apply_patterns.vector.lower_transfer
    # CHECK-SAME: max_transfer_rank = 4
    vector.ApplyLowerTransferPatternsOp(max_transfer_rank=4)
    # CHECK: transform.apply_patterns.vector.transfer_to_scf
    # CHECK-SAME: max_transfer_rank = 3
    # CHECK-SAME: full_unroll = true
    vector.ApplyTransferToScfPatternsOp(max_transfer_rank=3, full_unroll=True)


@run_apply_patterns
def enum_configurable_patterns():
    # CHECK: transform.apply_patterns.vector.lower_contraction
    vector.ApplyLowerContractionPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_contraction
    # CHECK-SAME: lowering_strategy = matmulintrinsics
    vector.ApplyLowerContractionPatternsOp(
        lowering_strategy=vector.VectorContractLowering.Matmul
    )
    # CHECK: transform.apply_patterns.vector.lower_contraction
    # CHECK-SAME: lowering_strategy = parallelarith
    vector.ApplyLowerContractionPatternsOp(
        lowering_strategy=vector.VectorContractLowering.ParallelArith
    )

    # CHECK: transform.apply_patterns.vector.lower_multi_reduction
    vector.ApplyLowerMultiReductionPatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_multi_reduction
    # This is the default mode, not printed.
    vector.ApplyLowerMultiReductionPatternsOp(
        lowering_strategy=vector.VectorMultiReductionLowering.InnerParallel
    )
    # CHECK: transform.apply_patterns.vector.lower_multi_reduction
    # CHECK-SAME: lowering_strategy = innerreduction
    vector.ApplyLowerMultiReductionPatternsOp(
        lowering_strategy=vector.VectorMultiReductionLowering.InnerReduction
    )

    # CHECK: transform.apply_patterns.vector.lower_transpose
    vector.ApplyLowerTransposePatternsOp()
    # CHECK: transform.apply_patterns.vector.lower_transpose
    # This is the default strategy, not printed.
    vector.ApplyLowerTransposePatternsOp(
        lowering_strategy=vector.VectorTransposeLowering.EltWise
    )
    # CHECK: transform.apply_patterns.vector.lower_transpose
    # CHECK-SAME: lowering_strategy = flat_transpose
    vector.ApplyLowerTransposePatternsOp(
        lowering_strategy=vector.VectorTransposeLowering.Flat
    )
    # CHECK: transform.apply_patterns.vector.lower_transpose
    # CHECK-SAME: lowering_strategy = shuffle_1d
    vector.ApplyLowerTransposePatternsOp(
        lowering_strategy=vector.VectorTransposeLowering.Shuffle1D
    )
    # CHECK: transform.apply_patterns.vector.lower_transpose
    # CHECK-SAME: lowering_strategy = shuffle_16x16
    vector.ApplyLowerTransposePatternsOp(
        lowering_strategy=vector.VectorTransposeLowering.Shuffle16x16
    )
    # CHECK: transform.apply_patterns.vector.lower_transpose
    # CHECK-SAME: lowering_strategy = flat_transpose
    # CHECK-SAME: avx2_lowering_strategy = true
    vector.ApplyLowerTransposePatternsOp(
        lowering_strategy=vector.VectorTransposeLowering.Flat,
        avx2_lowering_strategy=True,
    )

    # CHECK: transform.apply_patterns.vector.split_transfer_full_partial
    vector.ApplySplitTransferFullPartialPatternsOp()
    # CHECK: transform.apply_patterns.vector.split_transfer_full_partial
    # CHECK-SAME: split_transfer_strategy = none
    vector.ApplySplitTransferFullPartialPatternsOp(
        split_transfer_strategy=vector.VectorTransferSplit.None_
    )
    # CHECK: transform.apply_patterns.vector.split_transfer_full_partial
    # CHECK-SAME: split_transfer_strategy = "vector-transfer"
    vector.ApplySplitTransferFullPartialPatternsOp(
        split_transfer_strategy=vector.VectorTransferSplit.VectorTransfer
    )
    # CHECK: transform.apply_patterns.vector.split_transfer_full_partial
    # This is the default mode, not printed.
    vector.ApplySplitTransferFullPartialPatternsOp(
        split_transfer_strategy=vector.VectorTransferSplit.LinalgCopy
    )
    # CHECK: transform.apply_patterns.vector.split_transfer_full_partial
    # CHECK-SAME: split_transfer_strategy = "force-in-bounds"
    vector.ApplySplitTransferFullPartialPatternsOp(
        split_transfer_strategy=vector.VectorTransferSplit.ForceInBounds
    )
