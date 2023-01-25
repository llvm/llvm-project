// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion))' -split-input-file | FileCheck %s
// RUN: mlir-opt -allow-unregistered-dialect %s -pass-pipeline='builtin.module(func.func(affine-loop-fusion{fusion-maximal}))' -split-input-file | FileCheck %s --check-prefix=MAXIMAL

// Part I of fusion tests in  mlir/test/Transforms/loop-fusion.mlir.
// Part II of fusion tests in mlir/test/Transforms/loop-fusion-2.mlir
// Part IV of fusion tests in mlir/test/Transforms/loop-fusion-4.mlir

// -----

// Test case from github bug 777.
// CHECK-LABEL: func @mul_add_0
func.func @mul_add_0(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32>, %arg2: memref<3x3xf32>, %arg3: memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<3x3xf32>
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 3 {
      affine.store %cst, %0[%arg4, %arg5] : memref<3x3xf32>
    }
  }
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 3 {
      affine.for %arg6 = 0 to 4 {
        %1 = affine.load %arg1[%arg6, %arg5] : memref<4x3xf32>
        %2 = affine.load %arg0[%arg4, %arg6] : memref<3x4xf32>
        %3 = arith.mulf %2, %1 : f32
        %4 = affine.load %0[%arg4, %arg5] : memref<3x3xf32>
        %5 = arith.addf %4, %3 : f32
        affine.store %5, %0[%arg4, %arg5] : memref<3x3xf32>
      }
    }
  }
  affine.for %arg4 = 0 to 3 {
    affine.for %arg5 = 0 to 3 {
      %6 = affine.load %arg2[%arg4, %arg5] : memref<3x3xf32>
      %7 = affine.load %0[%arg4, %arg5] : memref<3x3xf32>
      %8 = arith.addf %7, %6 : f32
      affine.store %8, %arg3[%arg4, %arg5] : memref<3x3xf32>
    }
  }
  // CHECK:      affine.for %[[i0:.*]] = 0 to 3 {
  // CHECK-NEXT:   affine.for %[[i1:.*]] = 0 to 3 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     affine.for %[[i2:.*]] = 0 to 4 {
  // CHECK-NEXT:       affine.load %{{.*}}[%[[i2]], %[[i1]]] : memref<4x3xf32>
  // CHECK-NEXT:       affine.load %{{.*}}[%[[i0]], %[[i2]]] : memref<3x4xf32>
  // CHECK-NEXT:       arith.mulf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:       affine.load %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:       arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:       affine.store %{{.*}}, %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     }
  // CHECK-NEXT:     affine.load %{{.*}}[%[[i0]], %[[i1]]] : memref<3x3xf32>
  // CHECK-NEXT:     affine.load %{{.*}}[0, 0] : memref<1x1xf32>
  // CHECK-NEXT:     arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:     affine.store %{{.*}}, %{{.*}}[%[[i0]], %[[i1]]] : memref<3x3xf32>
  // CHECK-NEXT:   }
  // CHECK-NEXT: }
  // CHECK-NEXT: return
  return
}

// -----

// Verify that 'fuseProducerConsumerNodes' fuse a producer loop with a store
// that has multiple outgoing edges.

// CHECK-LABEL: func @should_fuse_multi_outgoing_edge_store_producer
func.func @should_fuse_multi_outgoing_edge_store_producer(%a : memref<1xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg0 = 0 to 1 {
    affine.store %cst, %a[%arg0] : memref<1xf32>
  }

  affine.for %arg0 = 0 to 1 {
    %0 = affine.load %a[%arg0] : memref<1xf32>
  }

  affine.for %arg0 = 0 to 1 {
    %0 = affine.load %a[%arg0] : memref<1xf32>
  }
  // CHECK:      affine.for %{{.*}} = 0 to 1 {
  // CHECK-NEXT:   affine.store
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT:   affine.load
  // CHECK-NEXT: }

  return
}

// -----

// Verify that 'fuseProducerConsumerNodes' fuses a producer loop that: 1) has
// multiple outgoing edges, 2) producer store has a single outgoing edge.
// Sibling loop fusion should not fuse any of these loops due to
// dependencies on external memrefs '%a' and '%b'.

// CHECK-LABEL: func @should_fuse_producer_with_multi_outgoing_edges
func.func @should_fuse_producer_with_multi_outgoing_edges(%a : memref<1xf32>, %b : memref<1xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg0 = 0 to 1 {
    %0 = affine.load %a[%arg0] : memref<1xf32>
    affine.store %cst, %b[%arg0] : memref<1xf32>
  }

  affine.for %arg0 = 0 to 1 {
    affine.store %cst, %a[%arg0] : memref<1xf32>
    %1 = affine.load %b[%arg0] : memref<1xf32>
  }
  // CHECK: affine.for %{{.*}} = 0 to 1
  // CHECK-NEXT: affine.load %[[A:.*]][{{.*}}]
  // CHECK-NEXT: affine.store %{{.*}}, %[[B:.*]][{{.*}}]
  // CHECK-NEXT: affine.store %{{.*}}, %[[A]]
  // CHECK-NEXT: affine.load %[[B]]
  // CHECK-NOT: affine.for %{{.*}}
  // CHECK: return
  return
}

// MAXIMAL-LABEL: func @reshape_into_matmul
func.func @reshape_into_matmul(%lhs : memref<1024x1024xf32>,
              %R: memref<16x64x1024xf32>, %out: memref<1024x1024xf32>) {
  %rhs = memref.alloc() :  memref<1024x1024xf32>

  // Reshape from 3-d to 2-d.
  affine.for %i0 = 0 to 16 {
    affine.for %i1 = 0 to 64 {
      affine.for %k = 0 to 1024 {
        %v = affine.load %R[%i0, %i1, %k] : memref<16x64x1024xf32>
        affine.store %v, %rhs[64*%i0 + %i1, %k] : memref<1024x1024xf32>
      }
    }
  }

  // Matmul.
  affine.for %i = 0 to 1024 {
    affine.for %j = 0 to 1024 {
      affine.for %k = 0 to 1024 {
        %0 = affine.load %rhs[%k, %j] : memref<1024x1024xf32>
        %1 = affine.load %lhs[%i, %k] : memref<1024x1024xf32>
        %2 = arith.mulf %1, %0 : f32
        %3 = affine.load %out[%i, %j] : memref<1024x1024xf32>
        %4 = arith.addf %3, %2 : f32
        affine.store %4, %out[%i, %j] : memref<1024x1024xf32>
      }
    }
  }
  return
}
// MAXIMAL-NEXT: memref.alloc
// MAXIMAL-NEXT: affine.for
// MAXIMAL-NEXT:   affine.for
// MAXIMAL-NEXT:     affine.for
// MAXIMAL-NOT:      affine.for
// MAXIMAL:      return

// -----

// CHECK-LABEL: func @vector_loop
func.func @vector_loop(%a : memref<10x20xf32>, %b : memref<10x20xf32>,
                  %c : memref<10x20xf32>) {
  affine.for %j = 0 to 10 {
    affine.for %i = 0 to 5 {
      %ld0 = affine.vector_load %a[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
      affine.vector_store %ld0, %b[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
    }
  }

  affine.for %j = 0 to 10 {
    affine.for %i = 0 to 5 {
      %ld0 = affine.vector_load %b[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
      affine.vector_store %ld0, %c[%j, %i*4] : memref<10x20xf32>, vector<4xf32>
    }
  }

  return
}
// CHECK:      affine.for
// CHECK-NEXT:   affine.for
// CHECK-NEXT:     affine.vector_load
// CHECK-NEXT:     affine.vector_store
// CHECK-NEXT:     affine.vector_load
// CHECK-NEXT:     affine.vector_store
// CHECK-NOT:  affine.for

// -----

// CHECK-LABEL: func @multi_outgoing_edges
func.func @multi_outgoing_edges(%in0 : memref<32xf32>,
                      %in1 : memref<32xf32>) {
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.addf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.subf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.mulf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.divf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  return
}

// CHECK:      affine.for
// CHECK-NOT:  affine.for
// CHECK:        arith.addf
// CHECK-NOT:  affine.for
// CHECK:        arith.subf
// CHECK-NOT:  affine.for
// CHECK:        arith.mulf
// CHECK-NOT:  affine.for
// CHECK:        arith.divf

// -----

// Test fusion when dynamically shaped memrefs are used with constant trip count loops.

// CHECK-LABEL: func @calc
func.func @calc(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %len: index) {
  %c1 = arith.constant 1 : index
  %1 = memref.alloc(%len) : memref<?xf32>
  affine.for %arg4 = 1 to 10 {
    %7 = affine.load %arg0[%arg4] : memref<?xf32>
    %8 = affine.load %arg1[%arg4] : memref<?xf32>
    %9 = arith.addf %7, %8 : f32
    affine.store %9, %1[%arg4] : memref<?xf32>
  }
  affine.for %arg4 = 1 to 10 {
    %7 = affine.load %1[%arg4] : memref<?xf32>
    %8 = affine.load %arg1[%arg4] : memref<?xf32>
    %9 = arith.mulf %7, %8 : f32
    affine.store %9, %arg2[%arg4] : memref<?xf32>
  }
  return
}
// CHECK:       memref.alloc() : memref<1xf32>
// CHECK:       affine.for %arg{{.*}} = 1 to 10 {
// CHECK-NEXT:    affine.load %arg{{.*}}
// CHECK-NEXT:    affine.load %arg{{.*}}
// CHECK-NEXT:    arith.addf
// CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
// CHECK-NEXT:    affine.load %{{.*}}[0] : memref<1xf32>
// CHECK-NEXT:    affine.load %arg{{.*}}[%arg{{.*}}] : memref<?xf32>
// CHECK-NEXT:    arith.mulf
// CHECK-NEXT:    affine.store %{{.*}}, %arg{{.*}}[%arg{{.*}}] : memref<?xf32>
// CHECK-NEXT:  }
// CHECK-NEXT:  return

// -----

// CHECK-LABEL: func @should_not_fuse_since_non_affine_users
func.func @should_not_fuse_since_non_affine_users(%in0 : memref<32xf32>,
                      %in1 : memref<32xf32>) {
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.addf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = memref.load %in0[%d] : memref<32xf32>
    %rhs = memref.load %in1[%d] : memref<32xf32>
    %add = arith.subf %lhs, %rhs : f32
    memref.store %add, %in0[%d] : memref<32xf32>
  }
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.mulf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  return
}

// CHECK:  affine.for
// CHECK:    arith.addf
// CHECK:  affine.for
// CHECK:    arith.subf
// CHECK:  affine.for
// CHECK:    arith.mulf

// -----

// CHECK-LABEL: func @should_not_fuse_since_top_level_non_affine_users
func.func @should_not_fuse_since_top_level_non_affine_users(%in0 : memref<32xf32>,
                      %in1 : memref<32xf32>) {
  %sum = memref.alloc() : memref<f32>
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.addf %lhs, %rhs : f32
    memref.store %add, %sum[] : memref<f32>
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  %load_sum = memref.load %sum[] : memref<f32>
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.mulf %lhs, %rhs : f32
    %sub = arith.subf %add, %load_sum: f32
    affine.store %sub, %in0[%d] : memref<32xf32>
  }
  memref.dealloc %sum : memref<f32>
  return
}

// CHECK:  affine.for
// CHECK:    arith.addf
// CHECK:  affine.for
// CHECK:    arith.mulf
// CHECK:    arith.subf

// -----

// CHECK-LABEL: func @should_not_fuse_since_top_level_non_affine_mem_write_users
func.func @should_not_fuse_since_top_level_non_affine_mem_write_users(
    %in0 : memref<32xf32>, %in1 : memref<32xf32>) {
  %c0 = arith.constant 0 : index
  %cst_0 = arith.constant 0.000000e+00 : f32

  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.addf %lhs, %rhs : f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  memref.store %cst_0, %in0[%c0] : memref<32xf32>
  affine.for %d = 0 to 32 {
    %lhs = affine.load %in0[%d] : memref<32xf32>
    %rhs = affine.load %in1[%d] : memref<32xf32>
    %add = arith.addf %lhs, %rhs: f32
    affine.store %add, %in0[%d] : memref<32xf32>
  }
  return
}

// CHECK:  affine.for
// CHECK:    arith.addf
// CHECK:  affine.for
// CHECK:    arith.addf

// -----

// MAXIMAL-LABEL: func @fuse_minor_affine_map
func.func @fuse_minor_affine_map(%in: memref<128xf32>, %out: memref<20x512xf32>) {
  %tmp = memref.alloc() : memref<128xf32>

  affine.for %arg4 = 0 to 128 {
    %ld = affine.load %in[%arg4] : memref<128xf32>
    affine.store %ld, %tmp[%arg4] : memref<128xf32>
  }

  affine.for %arg3 = 0 to 20 {
    affine.for %arg4 = 0 to 512 {
      %ld = affine.load %tmp[%arg4 mod 128] : memref<128xf32>
      affine.store %ld, %out[%arg3, %arg4] : memref<20x512xf32>
    }
  }

  return
}

// TODO: The size of the private memref is not properly computed in the presence
// of the 'mod' operation. It should be memref<1xf32> instead of
// memref<128xf32>: https://bugs.llvm.org/show_bug.cgi?id=46973
// MAXIMAL:       memref.alloc() : memref<128xf32>
// MAXIMAL:       affine.for
// MAXIMAL-NEXT:    affine.for
// MAXIMAL-NOT:   affine.for
// MAXIMAL:       return

// -----

// CHECK-LABEL: func @should_fuse_multi_store_producer_and_privatize_memfefs
func.func @should_fuse_multi_store_producer_and_privatize_memfefs() {
  %a = memref.alloc() : memref<10xf32>
  %b = memref.alloc() : memref<10xf32>
  %c = memref.alloc() : memref<10xf32>
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %arg0 = 0 to 10 {
    affine.store %cst, %a[%arg0] : memref<10xf32>
    affine.store %cst, %b[%arg0] : memref<10xf32>
    affine.store %cst, %c[%arg0] : memref<10xf32>
    %0 = affine.load %c[%arg0] : memref<10xf32>
  }

  affine.for %arg0 = 0 to 10 {
    %0 = affine.load %a[%arg0] : memref<10xf32>
  }

  affine.for %arg0 = 0 to 10 {
    %0 = affine.load %b[%arg0] : memref<10xf32>
  }

	// All the memrefs should be privatized except '%c', which is not involved in
  // the producer-consumer fusion.
  // CHECK:      affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT:   affine.load %{{.*}}[0] : memref<1xf32>
  // CHECK-NEXT: }

  return
}


func.func @should_fuse_multi_store_producer_with_escaping_memrefs_and_remove_src(
    %a : memref<10xf32>, %b : memref<10xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
    affine.store %cst, %b[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 10 {
    %0 = affine.load %a[%i1] : memref<10xf32>
  }

  affine.for %i2 = 0 to 10 {
    %0 = affine.load %b[%i2] : memref<10xf32>
  }

	// Producer loop '%i0' should be removed after fusion since fusion is maximal.
  // No memref should be privatized since they escape the function, and the
  // producer is removed after fusion.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for

  return
}

// -----

func.func @should_fuse_multi_store_producer_with_escaping_memrefs_and_preserve_src(
    %a : memref<10xf32>, %b : memref<10xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
    affine.store %cst, %b[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 5 {
    %0 = affine.load %a[%i1] : memref<10xf32>
  }

  affine.for %i2 = 0 to 10 {
    %0 = affine.load %b[%i2] : memref<10xf32>
  }

	// Loops '%i0' and '%i2' should be fused first and '%i0' should be removed
  // since fusion is maximal. Then the fused loop and '%i1' should be fused
  // and the fused loop shouldn't be removed since fusion is not maximal.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK:       affine.for %{{.*}} = 0 to 5 {
  // CHECK-NEXT:    affine.store %{{.*}} : memref<1xf32>
  // CHECK-NEXT:    affine.store %{{.*}} : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}} : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}} : memref<1xf32>
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for

  return
}


func.func @should_not_fuse_due_to_dealloc(%arg0: memref<16xf32>){
  %A = memref.alloc() : memref<16xf32>
  %C = memref.alloc() : memref<16xf32>
  %cst_1 = arith.constant 1.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %arg0[%arg1] : memref<16xf32>
    affine.store %a, %A[%arg1] : memref<16xf32>
    affine.store %a, %C[%arg1] : memref<16xf32>
  }
  memref.dealloc %C : memref<16xf32>
  %B = memref.alloc() : memref<16xf32>
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %A[%arg1] : memref<16xf32>
    %b = arith.addf %cst_1, %a : f32
    affine.store %b, %B[%arg1] : memref<16xf32>
  }
  memref.dealloc %A : memref<16xf32>
  return
}
// CHECK-LABEL: func @should_not_fuse_due_to_dealloc
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.store
// CHECK-NEXT:      affine.store
// CHECK:         memref.dealloc
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      arith.addf
// CHECK-NEXT:      affine.store

// -----

// CHECK-LABEL: func @should_fuse_defining_node_has_no_dependence_from_source_node
func.func @should_fuse_defining_node_has_no_dependence_from_source_node(
    %a : memref<10xf32>, %b : memref<f32>) -> () {
  affine.for %i0 = 0 to 10 {
    %0 = affine.load %b[] : memref<f32>
    affine.store %0, %a[%i0] : memref<10xf32>
  }
  %0 = affine.load %b[] : memref<f32>
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %a[%i1] : memref<10xf32>
    %2 = arith.divf %0, %1 : f32
  }

  // Loops '%i0' and '%i1' should be fused even though there is a defining node
  // between the loops. It is because the node has no dependence from '%i0'.
  // CHECK:       affine.load %{{.*}}[] : memref<f32>
  // CHECK-NEXT:  affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[] : memref<f32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    arith.divf
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_defining_node_has_dependence_from_source_loop
func.func @should_not_fuse_defining_node_has_dependence_from_source_loop(
    %a : memref<10xf32>, %b : memref<f32>) -> () {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %b[] : memref<f32>
    affine.store %cst, %a[%i0] : memref<10xf32>
  }
  %0 = affine.load %b[] : memref<f32>
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %a[%i1] : memref<10xf32>
    %2 = arith.divf %0, %1 : f32
  }

  // Loops '%i0' and '%i1' should not be fused because the defining node of '%0'
  // used in '%i1' has dependence from loop '%i0'.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[] : memref<f32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.load %{{.*}}[] : memref<f32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    arith.divf
  // CHECK-NEXT:  }
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_defining_node_has_transitive_dependence_from_source_loop
func.func @should_not_fuse_defining_node_has_transitive_dependence_from_source_loop(
    %a : memref<10xf32>, %b : memref<10xf32>, %c : memref<f32>) -> () {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
    affine.store %cst, %b[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %b[%i1] : memref<10xf32>
    affine.store %1, %c[] : memref<f32>
  }
  %0 = affine.load %c[] : memref<f32>
  affine.for %i2 = 0 to 10 {
    %1 = affine.load %a[%i2] : memref<10xf32>
    %2 = arith.divf %0, %1 : f32
  }

	// When loops '%i0' and '%i2' are evaluated first, they should not be
  // fused. The defining node of '%0' in loop '%i2' has transitive dependence
  // from loop '%i0'. After that, loops '%i0' and '%i1' are evaluated, and they
  // will be fused as usual.
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[] : memref<f32>
  // CHECK-NEXT:  }
  // CHECK-NEXT:  affine.load %{{.*}}[] : memref<f32>
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    arith.divf
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for
  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_dest_loop_nest_return_value
func.func @should_not_fuse_dest_loop_nest_return_value(
    %a : memref<10xf32>) -> () {
  %cst = arith.constant 0.000000e+00 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %a[%i0] : memref<10xf32>
  }
  %b = affine.for %i1 = 0 to 10 step 2 iter_args(%b_iter = %cst) -> f32 {
    %load_a = affine.load %a[%i1] : memref<10xf32>
    affine.yield %load_a: f32
  }

  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK:       affine.for %{{.*}} = 0 to 10 step 2 iter_args(%{{.*}} = %{{.*}}) -> (f32) {
  // CHECK-NEXT:    affine.load
  // CHECK-NEXT:    affine.yield
  // CHECK-NEXT:  }

  return
}

// -----

// CHECK-LABEL: func @should_not_fuse_src_loop_nest_return_value
func.func @should_not_fuse_src_loop_nest_return_value(
    %a : memref<10xf32>) -> () {
  %cst = arith.constant 1.000000e+00 : f32
  %b = affine.for %i = 0 to 10 step 2 iter_args(%b_iter = %cst) -> f32 {
    %c = arith.addf %b_iter, %b_iter : f32
    affine.store %c, %a[%i] : memref<10xf32>
    affine.yield %c: f32
  }
  affine.for %i1 = 0 to 10 {
    %1 = affine.load %a[%i1] : memref<10xf32>
  }

  // CHECK:       %{{.*}} = affine.for %{{.*}} = 0 to 10 step 2 iter_args(%{{.*}} = %{{.*}}) -> (f32) {
  // CHECK-NEXT:    %{{.*}} = arith.addf %{{.*}}, %{{.*}} : f32
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.yield %{{.*}} : f32
  // CHECK-NEXT:  }
  // CHECK:       affine.for %{{.*}} = 0 to 10 {
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }

  return
}

// -----

func.func private @some_function(memref<16xf32>)
func.func @call_op_prevents_fusion(%arg0: memref<16xf32>){
  %A = memref.alloc() : memref<16xf32>
  %cst_1 = arith.constant 1.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %arg0[%arg1] : memref<16xf32>
    affine.store %a, %A[%arg1] : memref<16xf32>
  }
  call @some_function(%A) : (memref<16xf32>) -> ()
  %B = memref.alloc() : memref<16xf32>
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %A[%arg1] : memref<16xf32>
    %b = arith.addf %cst_1, %a : f32
    affine.store %b, %B[%arg1] : memref<16xf32>
  }
  return
}
// CHECK-LABEL: func @call_op_prevents_fusion
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.store
// CHECK:         call
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      arith.addf
// CHECK-NEXT:      affine.store

// -----

func.func private @some_function()
func.func @call_op_does_not_prevent_fusion(%arg0: memref<16xf32>){
  %A = memref.alloc() : memref<16xf32>
  %cst_1 = arith.constant 1.000000e+00 : f32
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %arg0[%arg1] : memref<16xf32>
    affine.store %a, %A[%arg1] : memref<16xf32>
  }
  call @some_function() : () -> ()
  %B = memref.alloc() : memref<16xf32>
  affine.for %arg1 = 0 to 16 {
    %a = affine.load %A[%arg1] : memref<16xf32>
    %b = arith.addf %cst_1, %a : f32
    affine.store %b, %B[%arg1] : memref<16xf32>
  }
  return
}
// CHECK-LABEL: func @call_op_does_not_prevent_fusion
// CHECK:         affine.for
// CHECK-NOT:     affine.for

// -----

// Test for source that writes to an escaping memref and has two consumers.
// Fusion should create private memrefs in place of `%arg0` since the source is
// not to be removed after fusion and the destinations do not write to `%arg0`.
// This should enable both the consumers to benefit from fusion, which would not
// be possible if private memrefs were not created.
func.func @should_fuse_with_both_consumers_separately(%arg0: memref<10xf32>) {
  %cf7 = arith.constant 7.0 : f32
  affine.for %i0 = 0 to 10 {
    affine.store %cf7, %arg0[%i0] : memref<10xf32>
  }
  affine.for %i1 = 0 to 7 {
    %v0 = affine.load %arg0[%i1] : memref<10xf32>
  }
  affine.for %i1 = 5 to 9 {
    %v0 = affine.load %arg0[%i1] : memref<10xf32>
  }
  return
}

// CHECK-LABEL: func @should_fuse_with_both_consumers_separately
// CHECK:         affine.for
// CHECK-NEXT:      affine.store
// CHECK:         affine.for
// CHECK-NEXT:      affine.store
// CHECK-NEXT:      affine.load
// CHECK:         affine.for
// CHECK-NEXT:      affine.store
// CHECK-NEXT:      affine.load

// -----

// Fusion is avoided when the slice computed is invalid. Comments below describe
// incorrect backward slice computation. Similar logic applies for forward slice
// as well.
func.func @no_fusion_cannot_compute_valid_slice() {
  %A = memref.alloc() : memref<5xf32>
  %B = memref.alloc() : memref<6xf32>
  %C = memref.alloc() : memref<5xf32>
  %cst = arith.constant 0. : f32

  affine.for %arg0 = 0 to 5 {
    %a = affine.load %A[%arg0] : memref<5xf32>
    affine.store %a, %B[%arg0 + 1] : memref<6xf32>
  }

  affine.for %arg0 = 0 to 5 {
    // Backward slice computed will be:
    // slice ( src loop: 0, dst loop: 1, depth: 1 : insert point: (1, 0)
    // loop bounds: [(d0) -> (d0 - 1), (d0) -> (d0)] )

    // Resulting fusion would be as below. It is easy to note the out-of-bounds
    // access by 'affine.load'.

    // #map0 = affine_map<(d0) -> (d0 - 1)>
    // #map1 = affine_map<(d0) -> (d0)>
    // affine.for %arg1 = #map0(%arg0) to #map1(%arg0) {
    //   %5 = affine.load %1[%arg1] : memref<5xf32>
    //   ...
    //   ...
    // }

    %a = affine.load %B[%arg0] : memref<6xf32>
    %b = arith.mulf %a, %cst : f32
    affine.store %b, %C[%arg0] : memref<5xf32>
  }
  return
}
// CHECK-LABEL: func @no_fusion_cannot_compute_valid_slice
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      affine.store
// CHECK:         affine.for
// CHECK-NEXT:      affine.load
// CHECK-NEXT:      arith.mulf
// CHECK-NEXT:      affine.store

// MAXIMAL-LABEL:   func @reduce_add_f32_f32(
func.func @reduce_add_f32_f32(%arg0: memref<64x64xf32, 1>, %arg1: memref<1x64xf32, 1>, %arg2: memref<1x64xf32, 1>) {
  %cst_0 = arith.constant 0.000000e+00 : f32
  %cst_1 = arith.constant 1.000000e+00 : f32
  %0 = memref.alloca() : memref<f32, 1>
  %1 = memref.alloca() : memref<f32, 1>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 64 {
      %accum = affine.for %arg5 = 0 to 64 iter_args (%prevAccum = %cst_0) -> f32 {
        %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
        %5 = arith.addf %prevAccum, %4 : f32
        affine.yield %5 : f32
      }
      %accum_dbl = arith.addf %accum, %accum : f32
      affine.store %accum_dbl, %arg1[%arg3, %arg4] : memref<1x64xf32, 1>
    }
  }
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 64 {
      %accum = affine.for %arg5 = 0 to 64 iter_args (%prevAccum = %cst_1) -> f32 {
        %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
        %5 = arith.mulf %prevAccum, %4 : f32
        affine.yield %5 : f32
      }
      %accum_sqr = arith.mulf %accum, %accum : f32
      affine.store %accum_sqr, %arg2[%arg3, %arg4] : memref<1x64xf32, 1>
    }
  }
  return
}
// The two loops here get maximally sibling-fused at the innermost
// insertion point. Test checks  if the innermost reduction loop of the fused loop
// gets promoted into its outerloop.
// MAXIMAL-SAME:                             %[[arg_0:.*]]: memref<64x64xf32, 1>,
// MAXIMAL-SAME:                             %[[arg_1:.*]]: memref<1x64xf32, 1>,
// MAXIMAL-SAME:                             %[[arg_2:.*]]: memref<1x64xf32, 1>) {
// MAXIMAL:             %[[cst:.*]] = arith.constant 0 : index
// MAXIMAL-NEXT:        %[[cst_0:.*]] = arith.constant 0.000000e+00 : f32
// MAXIMAL-NEXT:        %[[cst_1:.*]] = arith.constant 1.000000e+00 : f32
// MAXIMAL:             affine.for %[[idx_0:.*]] = 0 to 1 {
// MAXIMAL-NEXT:          affine.for %[[idx_1:.*]] = 0 to 64 {
// MAXIMAL-NEXT:            %[[results:.*]]:2 = affine.for %[[idx_2:.*]] = 0 to 64 iter_args(%[[iter_0:.*]] = %[[cst_1]], %[[iter_1:.*]] = %[[cst_0]]) -> (f32, f32) {
// MAXIMAL-NEXT:              %[[val_0:.*]] = affine.load %[[arg_0]][%[[idx_2]], %[[idx_1]]] : memref<64x64xf32, 1>
// MAXIMAL-NEXT:              %[[reduc_0:.*]] = arith.addf %[[iter_1]], %[[val_0]] : f32
// MAXIMAL-NEXT:              %[[val_1:.*]] = affine.load %[[arg_0]][%[[idx_2]], %[[idx_1]]] : memref<64x64xf32, 1>
// MAXIMAL-NEXT:              %[[reduc_1:.*]] = arith.mulf %[[iter_0]], %[[val_1]] : f32
// MAXIMAL-NEXT:              affine.yield %[[reduc_1]], %[[reduc_0]] : f32, f32
// MAXIMAL-NEXT:            }
// MAXIMAL-NEXT:            %[[reduc_0_dbl:.*]] = arith.addf %[[results:.*]]#1, %[[results]]#1 : f32
// MAXIMAL-NEXT:            affine.store %[[reduc_0_dbl]], %[[arg_1]][%[[cst]], %[[idx_1]]] : memref<1x64xf32, 1>
// MAXIMAL-NEXT:            %[[reduc_1_sqr:.*]] = arith.mulf %[[results]]#0, %[[results]]#0 : f32
// MAXIMAL-NEXT:            affine.store %[[reduc_1_sqr]], %[[arg_2]][%[[idx_0]], %[[idx_1]]] : memref<1x64xf32, 1>
// MAXIMAL-NEXT:          }
// MAXIMAL-NEXT:        }
// MAXIMAL-NEXT:        return
// MAXIMAL-NEXT:      }

// -----

// CHECK-LABEL:   func @reduce_add_non_innermost
func.func @reduce_add_non_innermost(%arg0: memref<64x64xf32, 1>, %arg1: memref<1x64xf32, 1>, %arg2: memref<1x64xf32, 1>) {
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %0 = memref.alloca() : memref<f32, 1>
  %1 = memref.alloca() : memref<f32, 1>
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 64 {
      %accum = affine.for %arg5 = 0 to 64 iter_args (%prevAccum = %cst) -> f32 {
        %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
        %5 = arith.addf %prevAccum, %4 : f32
        affine.yield %5 : f32
      }
      %accum_dbl = arith.addf %accum, %accum : f32
      affine.store %accum_dbl, %arg1[%arg3, %arg4] : memref<1x64xf32, 1>
    }
  }
  affine.for %arg3 = 0 to 1 {
    affine.for %arg4 = 0 to 64 {
      %accum = affine.for %arg5 = 0 to 64 iter_args (%prevAccum = %cst_0) -> f32 {
        %4 = affine.load %arg0[%arg5, %arg4] : memref<64x64xf32, 1>
        %5 = arith.mulf %prevAccum, %4 : f32
        affine.yield %5 : f32
      }
      %accum_sqr = arith.mulf %accum, %accum : f32
      affine.store %accum_sqr, %arg2[%arg3, %arg4] : memref<1x64xf32, 1>
    }
  }
  return
}
// Test checks the loop structure is preserved after sibling fusion.
// CHECK:         affine.for
// CHECK-NEXT:      affine.for
// CHECK-NEXT:        affine.for
// CHECK             affine.for



// -----

// CHECK-LABEL: func @fuse_large_number_of_loops
func.func @fuse_large_number_of_loops(%arg0: memref<20x10xf32, 1>, %arg1: memref<20x10xf32, 1>, %arg2: memref<20x10xf32, 1>, %arg3: memref<20x10xf32, 1>, %arg4: memref<20x10xf32, 1>, %arg5: memref<f32, 1>, %arg6: memref<f32, 1>, %arg7: memref<f32, 1>, %arg8: memref<f32, 1>, %arg9: memref<20x10xf32, 1>, %arg10: memref<20x10xf32, 1>, %arg11: memref<20x10xf32, 1>, %arg12: memref<20x10xf32, 1>) {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = memref.alloc() : memref<f32, 1>
  affine.store %cst, %0[] : memref<f32, 1>
  %1 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg6[] : memref<f32, 1>
      affine.store %21, %1[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %2 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %1[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %arg3[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %2[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %3 = memref.alloc() : memref<f32, 1>
  %4 = affine.load %arg6[] : memref<f32, 1>
  %5 = affine.load %0[] : memref<f32, 1>
  %6 = arith.subf %5, %4 : f32
  affine.store %6, %3[] : memref<f32, 1>
  %7 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %3[] : memref<f32, 1>
      affine.store %21, %7[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %8 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg1[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %7[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %8[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %9 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg1[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %8[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %9[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %9[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %2[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.addf %22, %21 : f32
      affine.store %23, %arg11[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %10 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %1[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %arg2[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %10[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %8[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %10[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.addf %22, %21 : f32
      affine.store %23, %arg10[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %11 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg10[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %arg10[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %11[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %12 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %11[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %arg11[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.subf %22, %21 : f32
      affine.store %23, %12[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %13 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg7[] : memref<f32, 1>
      affine.store %21, %13[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %14 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg4[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %13[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %14[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %15 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg8[] : memref<f32, 1>
      affine.store %21, %15[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %16 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %15[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %12[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.addf %22, %21 : f32
      affine.store %23, %16[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %17 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %16[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = math.sqrt %21 : f32
      affine.store %22, %17[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %18 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg5[] : memref<f32, 1>
      affine.store %21, %18[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %19 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg1[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %18[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.mulf %22, %21 : f32
      affine.store %23, %19[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  %20 = memref.alloc() : memref<20x10xf32, 1>
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %17[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %19[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.divf %22, %21 : f32
      affine.store %23, %20[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %20[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %14[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.addf %22, %21 : f32
      affine.store %23, %arg12[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  affine.for %arg13 = 0 to 20 {
    affine.for %arg14 = 0 to 10 {
      %21 = affine.load %arg12[%arg13, %arg14] : memref<20x10xf32, 1>
      %22 = affine.load %arg0[%arg13, %arg14] : memref<20x10xf32, 1>
      %23 = arith.subf %22, %21 : f32
      affine.store %23, %arg9[%arg13, %arg14] : memref<20x10xf32, 1>
    }
  }
  return
}
// CHECK:         affine.for
// CHECK:         affine.for
// CHECK-NOT:     affine.for

// CHECK-LABEL: func @alias_escaping_memref
func.func @alias_escaping_memref(%a : memref<2x5xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alias = memref.reinterpret_cast %a to offset: [0], sizes: [10], strides: [1] : memref<2x5xf32> to memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %alias[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 10 {
    %0 = affine.load %alias[%i1] : memref<10xf32>
  }
  // Fusion happens, but memref isn't privatized since %alias is an alias of a
  // function argument.
  // CHECK:       memref.reinterpret_cast
  // CHECK-NEXT:  affine.for
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for

  return
}

// CHECK-LABEL: func @unknown_memref_def_op
func.func @unknown_memref_def_op() {
  %cst = arith.constant 0.000000e+00 : f32
  %may_alias = call @bar() : () -> memref<10xf32>
  affine.for %i0 = 0 to 10 {
    affine.store %cst, %may_alias[%i0] : memref<10xf32>
  }

  affine.for %i1 = 0 to 10 {
    %0 = affine.load %may_alias[%i1] : memref<10xf32>
  }
  // Fusion happens, but memref isn't privatized since %may_alias's origin is
  // unknown.
  // CHECK:       call
  // CHECK-NEXT:  affine.for
  // CHECK-NEXT:    affine.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:    affine.load %{{.*}}[%{{.*}}] : memref<10xf32>
  // CHECK-NEXT:  }
  // CHECK-NOT:   affine.for

  return
}
func.func private @bar() -> memref<10xf32>


// Add further tests in mlir/test/Transforms/loop-fusion-4.mlir
