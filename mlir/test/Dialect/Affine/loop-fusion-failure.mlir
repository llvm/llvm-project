// RUN: mlir-opt -allow-unregistered-dialect %s -test-loop-fusion=test-loop-fusion-failure -split-input-file -verify-diagnostics
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{mode=producer maximal}))' -split-input-file | FileCheck %s --check-prefix=PRODUCTION

#dynamic_index = affine_map<()[s0, s1] -> (s0 * s1)>

// An unsupported same-memref access relation must stop producer-consumer
// fusion at the dependence check. The destination loops remain separate.

// The utility-level check above proves that an inconclusive dependence is
// reported before slice computation. This run exercises the production pass:
// the same case must retain both two-dimensional loop nests and their order.
// PRODUCTION-LABEL: func.func @failed_dependence_fusion(
// PRODUCTION:       affine.for %{{.*}} = 1 to 8 {
// PRODUCTION-NEXT:    affine.for %{{.*}} = 1 to 8 {
// PRODUCTION-NEXT:      %{{.*}} = affine.apply
// PRODUCTION-NEXT:      affine.store
// PRODUCTION-NEXT:    }
// PRODUCTION-NEXT:  }
// PRODUCTION-NEXT:  affine.for %{{.*}} = 1 to 8 {
// PRODUCTION-NEXT:    affine.for %{{.*}} = 1 to 8 {
// PRODUCTION-NEXT:      %{{.*}} = affine.apply
// PRODUCTION-NEXT:      %{{.*}} = affine.load
// PRODUCTION-NEXT:      affine.store
// PRODUCTION-NEXT:    }
// PRODUCTION-NEXT:  }
// PRODUCTION-NEXT:  return
func.func @failed_dependence_fusion(
    %A: memref<?x9x9xi32>, %p: index, %q: index, %value: i32) {
  affine.for %i = 1 to 8 {
    // expected-remark@-1 {{fusion dependence prevents fusion}}
    affine.for %j = 1 to 8 {
      %z = affine.apply #dynamic_index()[%p, %q]
      affine.store %value, %A[%z, %i, %j] : memref<?x9x9xi32>
    }
  }
  affine.for %i = 1 to 8 {
    affine.for %j = 1 to 8 {
      %z = affine.apply #dynamic_index()[%p, %q]
      %loaded = affine.load %A[%z, %i - 1, %j + 1] : memref<?x9x9xi32>
      affine.store %loaded, %A[%z, %i - 1, %j + 1] : memref<?x9x9xi32>
    }
  }
  return
}
