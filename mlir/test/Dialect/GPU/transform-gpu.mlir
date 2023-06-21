// RUN: mlir-opt --test-transform-dialect-interpreter --split-input-file  -canonicalize -cse %s | FileCheck %s

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy2dblock(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGT:[0-9a-z]+]]: memref<32xf32>
func.func @saxpy2dblock(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %one = arith.constant 1 : index
//      CHECK:   gpu.launch
//      CHECK:   %[[BLKX:.*]] = gpu.block_id  x
//      CHECK:   %[[BLKY:.*]] = gpu.block_id  y
//      CHECK:   memref.load %[[ARGX]][%[[BLKX]], %[[BLKY]]]
//      CHECK:   memref.load %[[ARGY]][%[[BLKX]], %[[BLKY]]]
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.block<x>, #gpu.block<y>]}
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_forall_to_blocks %funcop grid_dims = [12, 9, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy2d(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<2x32xf32>
// CHECK-SAME:    %[[ARGT:[0-9a-z]+]]: memref<32xf32>
func.func @saxpy2d(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %one = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
//      CHECK:   %[[C1:.*]] = arith.constant 1 : index
//      CHECK:   %[[C12:.*]] = arith.constant 12 : index
//      CHECK:   %[[C9:.*]] = arith.constant 9 : index
//      CHECK:   %[[C7:.*]] = arith.constant 7 : index
//      CHECK:   gpu.launch async [%{{.*}}] blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C1]], %{{.*}} = %[[C1]], %{{.*}} = %[[C1]]) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C12]], %{{.*}} = %[[C9]], %{{.*}} = %[[C1]])
//      CHECK:   %[[TIDX:.*]] = gpu.thread_id  x
//      CHECK:   %[[TIDY:.*]] = gpu.thread_id  y
//      CHECK:   arith.cmpi ult, %[[TIDX]], %[[C9]] : index
//      CHECK:   arith.cmpi ult, %[[TIDY]], %[[C7]] : index
//      CHECK:   memref.load %[[ARGX]][%[[TIDY]], %[[TIDX]]]
//      CHECK:   memref.load %[[ARGY]][%[[TIDY]], %[[TIDX]]]
//      CHECK:   gpu.barrier
//      CHECK:   arith.cmpi ult, %[[TIDY]], %[[C1]] : index
//      CHECK:   memref.load %[[ARGT]][%[[TIDX]]]
//      CHECK:   gpu.barrier
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>]}
     scf.forall (%i) in (%c12) {
        %7 = memref.load %t[%i] : !type1d
        %8 = arith.addf %alpha, %7 : f32
        memref.store %8, %t[%i] : !type1d
     }  {mapping = [#gpu.thread<x>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [12, 9, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

!type4d = memref<32x64x4x32xf32>

// CHECK-LABEL: func.func @saxpy4d(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<32x64x4x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<32x64x4x32xf32>
func.func @saxpy4d(%x: !type4d, %y: !type4d, %alpha : f32) -> !type4d {
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %c4 = arith.constant 4 : index
//      CHECK:   %[[C32:.*]] = arith.constant 32 : index
//      CHECK:   %[[C64:.*]] = arith.constant 64 : index
//      CHECK:   %[[C4:.*]] = arith.constant 4 : index
//      CHECK:   %[[C1:.*]] = arith.constant 1 : index
//      CHECK:   gpu.launch blocks(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C32]], %{{.*}} = %[[C64]], %{{.*}} = %[[C1]]) threads(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}} = %[[C32]], %{{.*}} = %[[C4]], %{{.*}} = %[[C1]])
//      CHECK:   %[[BLKX:.*]] = gpu.block_id  x
//      CHECK:   %[[BLKY:.*]] = gpu.block_id  y
//      CHECK:   %[[TIDX:.*]] = gpu.thread_id  x
//      CHECK:   %[[TIDY:.*]] = gpu.thread_id  y
//      CHECK:   memref.load %[[ARGX]][%[[BLKX]], %[[BLKY]], %[[TIDY]], %[[TIDX]]]
//      CHECK:   memref.load %[[ARGY]][%[[BLKX]], %[[BLKY]], %[[TIDY]], %[[TIDX]]]
  scf.forall (%i, %j) in (%c32, %c64) {
    scf.forall (%k, %l) in (%c4, %c32) {
      %4 = memref.load %x[%i, %j, %k, %l] : !type4d
      %5 = memref.load %y[%i, %j, %k, %l] : !type4d
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j, %k, %l] : !type4d
    }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
  }  { mapping = [#gpu.block<x>, #gpu.block<y>] }
  return %y : !type4d
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["func.func"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %gpuLaunch = transform.gpu.map_forall_to_blocks %funcop { generate_gpu_launch } : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %gpuLaunch block_dims = [32, 4, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy2d_no_barrier(
func.func @saxpy2d_no_barrier(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %one = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
//  CHECK-NOT:   gpu.barrier
//      CHECK:   return
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j) in (%c7, %c9) {
        %4 = memref.load %x[%i, %j] : !type
        %5 = memref.load %y[%i, %j] : !type
        %6 = math.fma %alpha, %4, %5 : f32
        memref.store %6, %y[%i, %j] : !type
     }  { mapping = [#gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [12, 9, 1] sync_after_distribute = false : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<32x32xf32>
// CHECK-LABEL: func.func @saxpy2d_singleloop(
// CHECK-SAME:    %[[ARGX:[0-9a-z]+]]: memref<32x32xf32>
// CHECK-SAME:    %[[ARGY:[0-9a-z]+]]: memref<32x32xf32>
func.func @saxpy2d_singleloop(%x: !type, %y: !type, %stream : !gpu.async.token) -> !type {
  %c32 = arith.constant 32 : index
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
//      CHECK:   %[[TIDX:.*]] = gpu.thread_id  x
//      CHECK:   memref.load %[[ARGX]][%[[TIDX]], %[[TIDX]]]
//      CHECK:   memref.load %[[ARGY]][%[[TIDX]], %[[TIDX]]]
    scf.forall (%i) in (%c32) {
        %4 = memref.load %x[%i, %i] : !type
        %5 = memref.load %y[%i, %i] : !type
        %6 = arith.mulf %4, %5 : f32
        memref.store %6, %y[%i, %i] : !type
     }  { mapping = [#gpu.thread<x>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [32, 1, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<3 x 2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-LABEL: func.func @saxpy3d_fold_id_z(
func.func @saxpy3d_fold_id_z(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %one = arith.constant 1 : index
  %c12 = arith.constant 12 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
//  CHECK: %[[C0:.+]] = arith.constant 0 : index
//  CHECK-NOT:   gpu.thread_id  z
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    scf.forall (%i, %j, %k) in (%one, %c7, %c9) {
//      CHECK:   memref.load %{{.*}}[%[[C0]],
//      CHECK:   memref.load %{{.*}}[%[[C0]],
        %4 = memref.load %x[%i, %j, %k] : !type
        %5 = memref.load %y[%i, %j, %k] : !type
        %6 = math.fma %alpha, %4, %5 : f32
//      CHECK:   memref.store %{{.*}}, %{{.*}}[%[[C0]]
        memref.store %6, %y[%i, %j, %k] : !type
     }  { mapping = [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %funcop block_dims = [12, 9, 1] sync_after_distribute = false : (!transform.any_op) -> !transform.any_op
}

// -----

!type = memref<2 x 32 x f32>
!type1d = memref<32 x f32>

// CHECK-DAG: #[[$MAPWX:.*]] = affine_map<(d0, d1) -> (((d0 + d1 * 12) floordiv 32) mod 3)>
// CHECK-DAG: #[[$MAPWY:.*]] = affine_map<(d0, d1) -> ((((d0 + d1 * 12) floordiv 32) mod 6) floordiv 3)>

// CHECK-DAG: #[[$MAPLIN:.*]] = affine_map<(d0, d1) -> (d0 + d1 * 12)>
// CHECK-DAG: #[[$MAPLX:.*]] = affine_map<(d0, d1) -> ((d0 + d1 * 12) mod 10)>
// CHECK-DAG: #[[$MAPLY:.*]] = affine_map<(d0, d1) -> (((d0 + d1 * 12) mod 20) floordiv 10)>

// CHECK-LABEL: func.func @map_multi_level(
func.func @map_multi_level(%x: !type, %y: !type, %t: !type1d, %alpha : f32, %stream : !gpu.async.token) -> !type {
  %one = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %c7 = arith.constant 7 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index

  // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
  // CHECK-DAG: %[[C11:.*]] = arith.constant 11 : index
  // CHECK-DAG: %[[C12:.*]] = arith.constant 12 : index
  // CHECK-DAG: %[[C20:.*]] = arith.constant 20 : index

  // check that both the thread level and the warp level got distributed.
  //  CHECK-NOT: #gpu.thread
  //  CHECK-NOT: #gpu.warp
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    // CHECK-DAG: %[[TIDX:.*]] = gpu.thread_id  x
    // CHECK-DAG: %[[TIDY:.*]] = gpu.thread_id  y
    scf.forall (%i, %j) in (%c7, %c9) {
      %4 = memref.load %x[%i, %j] : !type
      %5 = memref.load %y[%i, %j] : !type
      %6 = math.fma %alpha, %4, %5 : f32
      memref.store %6, %y[%i, %j] : !type
    }  { mapping = [#gpu.thread<y>, #gpu.thread<x>]}

    // CHECK-DAG: %[[WIDX:.*]] = affine.apply #[[$MAPWX]](%[[TIDX]], %[[TIDY]])
    // CHECK-DAG: %[[WIDY:.*]] = affine.apply #[[$MAPWY]](%[[TIDX]], %[[TIDY]])
    // CHECK-DAG: %[[CMPX:.*]] = arith.cmpi ult, %[[WIDX]], %[[C1]] : index
    // CHECK-DAG: %[[CMPY:.*]] = arith.cmpi ult, %[[WIDY]], %[[C1]] : index
    //     CHECK: %[[COND:.*]] = arith.andi %[[CMPX]], %[[CMPY]] : i1
    //     CHECK: scf.if %[[COND]]
    scf.forall (%i) in (%c1) {
        %7 = memref.load %t[%i] : !type1d
        %8 = arith.addf %alpha, %7 : f32
        memref.store %8, %t[%i] : !type1d
     }  {mapping = [#gpu.warp<x>] }

    // CHECK-DAG: %[[LIN:.*]] = affine.apply #[[$MAPLIN]](%[[TIDX]], %[[TIDY]])
    // CHECK-DAG: %[[LIDX:.*]] = affine.apply #[[$MAPLX]](%[[TIDX]], %[[TIDY]])
    // CHECK-DAG: %[[LIDY:.*]] = affine.apply #[[$MAPLY]](%[[TIDX]], %[[TIDY]])
    // CHECK-DAG: %[[COND:.*]] = arith.cmpi ult, %[[LIN]], %[[C20]] : index
    //     CHECK: scf.if %[[COND]]
    //     CHECK:   memref.load %{{.*}}[%[[LIDX]]] : memref<32xf32>
    //     CHECK:   memref.store %{{.*}}[%[[LIDY]]] : memref<32xf32>
    scf.forall (%i, %j) in (%c10, %c2) {
        %7 = memref.load %t[%i] : !type1d
        %8 = arith.addf %alpha, %7 : f32
        memref.store %8, %t[%j] : !type1d
     }  {mapping = [#gpu.linear<x>, #gpu.linear<y>] }
    gpu.terminator
  }
  return %y : !type
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %funcop = transform.structured.match ops{["gpu.launch"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  transform.gpu.map_nested_forall_to_threads %funcop
    block_dims = [12, 11, 1] warp_dims = [3, 2, 1] : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: func.func @tiling_buffer_semantic_op(
//       CHECK:   gpu.launch {{.*}} {
//       CHECK:     scf.forall {{.*}} {
//       CHECK:       memref.subview
//       CHECK:       memref.subview
//       CHECK:       linalg.generic
//       CHECK:     }
//       CHECK:   }
func.func @tiling_buffer_semantic_op(%x: memref<32x32xf32>, %y: memref<32x32xf32>, %stream : !gpu.async.token) {
  %one = arith.constant 1 : index
  %name = gpu.launch async[%stream] blocks(%arg3, %arg4, %arg5) in (%arg9 = %one, %arg10 = %one, %arg11 = %one)
            threads(%arg6, %arg7, %arg8) in (%arg12 = %one, %arg13 = %one, %arg14 = %one)
  {
    linalg.generic
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]}
      ins(%x : memref<32x32xf32>)
      outs(%y : memref<32x32xf32>) {
        ^bb0(%in: f32, %out: f32):
          linalg.yield %in : f32
    }
    gpu.terminator
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.generic"]} in %arg0 : (!transform.any_op) -> !transform.any_op
  %forall, %tiled = transform.structured.tile_to_forall_op %matmul num_threads [10, 20, 30] (mapping = [ #gpu.thread<y>, #gpu.thread<x>, #gpu.thread<z> ] )
    : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
}
