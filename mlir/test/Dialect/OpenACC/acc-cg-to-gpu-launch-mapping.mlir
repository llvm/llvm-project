// RUN: mlir-opt %s --pass-pipeline="builtin.module(func.func(acc-cg-to-gpu))" | FileCheck %s

// ** tests for mapping par loops to gpu dimensions **

// CHECK-LABEL: @par0_loop
// CHECK: gpu.launch
// CHECK: %[[c1:.*]] = arith.constant 1 : index
// CHECK: %[[c4:.*]] = arith.constant 4 : index
// CHECK: scf.parallel (%[[iv:.*]]) = (%[[c1]]) to (%[[c4]]) step (%[[c1]])
func.func @par0_loop() {
  acc.compute_region {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%iv) = (%c1) to (%c4) step (%c1) {
      scf.reduce
    } {acc.par_dims = #acc<par_dims[sequential]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @par1_loop
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val:[a-z0-9_]+]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
// CHECK-NOT: scf.parallel
// CHECK: arith.addi %[[tidx]], %[[c4:.*]] : index
func.func @par1_loop() {
  acc.compute_region {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%iv) = (%c1) to (%c4) step (%c1) {
      %i = arith.addi %iv, %c4 : index
      scf.reduce
    } {acc.par_dims = #acc<par_dims[thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @par1_0_loop
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val:[a-z0-9_]+]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
// CHECK: scf.parallel (%[[iv2:.*]]) = (%[[c1:.*]]) to (%[[c4:.*]]) step (%[[c1]])
// CHECK: arith.addi %[[tidx]], %[[c1]]
// CHECK: arith.addi %[[iv2]], %[[c4]]
func.func @par1_0_loop() {
  acc.compute_region {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%iv1) = (%c1) to (%c4) step (%c1) {
      scf.parallel (%iv2) = (%c1) to (%c4) step (%c1) {
        %i1 = arith.addi %iv1, %c1 : index
        %i2 = arith.addi %iv2, %c4 : index
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[thread_x]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @par0_1_loop
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val:[a-z0-9_]+]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
// CHECK: scf.parallel (%[[iv1:.*]]) = (%[[c1:.*]]) to (%[[c4:.*]]) step (%[[c1]])
// CHECK: arith.addi %[[iv1]], %[[c1]]
// CHECK: arith.addi %[[tidx]], %[[c4]]
func.func @par0_1_loop() {
  acc.compute_region {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%iv1) = (%c1) to (%c4) step (%c1) {
      scf.parallel (%iv2) = (%c1) to (%c4) step (%c1) {
        %i1 = arith.addi %iv1, %c1 : index
        %i2 = arith.addi %iv2, %c4 : index
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[sequential]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @par2_1_loop
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val:[a-z0-9_]+]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
// CHECK-NOT: scf.parallel
// CHECK: arith.addi %[[tidy]], %[[c1:.*]] : index
// CHECK: arith.addi %[[tidx]], %[[c4:.*]] : index
func.func @par2_1_loop() {
  acc.compute_region {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%iv1) = (%c1) to (%c4) step (%c1) {
      scf.parallel (%iv2) = (%c1) to (%c4) step (%c1) {
        %i1 = arith.addi %iv1, %c1 : index
        %i2 = arith.addi %iv2, %c4 : index
        scf.reduce
      } {acc.par_dims = #acc<par_dims[thread_x]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[thread_y]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @par2_0_1_loop
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val:[a-z0-9_]+]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
// CHECK: arith.addi %[[tidy]], %[[c1:.*]] : index
// CHECK: scf.parallel (%[[iv2:.*]]) = (%[[c1]]) to (%[[c4:.*]]) step (%[[c1]])
// CHECK: arith.addi %[[iv2]], %[[tidx]]
func.func @par2_0_1_loop() {
  %par_dim1 = acc.par_width {par_dim = #acc.par_dim<thread_x>}
  %par_dim2 = acc.par_width {par_dim = #acc.par_dim<thread_y>}
  acc.compute_region launch(%arg0 = %par_dim1, %arg1 = %par_dim2) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    scf.parallel (%iv1) = (%c1) to (%c4) step (%c1) {
      %i1 = arith.addi %iv1, %c1 : index
      scf.parallel (%iv2) = (%c1) to (%c4) step (%c1) {
        scf.parallel (%iv3) = (%c1) to (%c4) step (%c1) {
          %i2 = arith.addi %iv2, %iv3 : index
          scf.reduce
        } {acc.par_dims = #acc<par_dims[thread_x]>}
        scf.reduce
      } {acc.par_dims = #acc<par_dims[sequential]>}
      scf.reduce
    } {acc.par_dims = #acc<par_dims[thread_y]>}
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// ** tests for launch arguments **

// CHECK-LABEL: @empty
// CHECK-NOT: acc.compute_region
// CHECK-NOT: acc.yield

// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val:[a-z0-9_]+]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
func.func @empty() {
  acc.compute_region {
  ^bb1:
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @empty_some_known_launch_arg

// CHECK: %[[bdimx_val:.*]] = arith.constant 32 : index
// CHECK-NOT: acc.compute_region
// CHECK-NOT: acc.yield

// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val:[a-z0-9_]+]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val:[a-z0-9_]+]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val:[a-z0-9_]+]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[bdimx_val]], %[[bdimy:[a-z0-9_]+]] = %[[bdimy_val:[a-z0-9_]+]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val:[a-z0-9_]+]])
func.func @empty_some_known_launch_arg() {
  %c32 = arith.constant 32 : index
  %par_dim1 = acc.par_width %c32 {par_dim = #acc.par_dim<thread_x>}
  acc.compute_region launch(%arg0 = %par_dim1) {
    acc.yield
  } {origin = "acc.parallel"}
  return
}


// CHECK-LABEL: @empty_all_known_launch_arg
// CHECK: %[[VL:.*]] = arith.constant 2 : index
// CHECK: %[[NW:.*]] = arith.constant 4 : index
// CHECK: %[[bdimz_val:.*]] = arith.constant 8 : index

// CHECK: %[[gdimx_val:.*]] = arith.constant 16 : index
// CHECK: %[[gdimy_val:.*]] = arith.constant 32 : index
// CHECK: %[[gdimz_val:.*]] = arith.constant 128 : index

// CHECK-NOT: acc.compute_region
// CHECK-NOT: acc.yield

// 2D block: blockDimX = VL, blockDimY = NW
// CHECK: gpu.launch
// CHECK-SAME: blocks(%[[bidx:[a-z0-9_]+]], %[[bidy:[a-z0-9_]+]], %[[bidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[gdimx:[a-z0-9_]+]] = %[[gdimx_val]], %[[gdimy:[a-z0-9_]+]] = %[[gdimy_val]], %[[gdimz:[a-z0-9_]+]] = %[[gdimz_val]])
// CHECK-SAME: threads(%[[tidx:[a-z0-9_]+]], %[[tidy:[a-z0-9_]+]], %[[tidz:[a-z0-9_]+]])
// CHECK-SAME: in (%[[bdimx:[a-z0-9_]+]] = %[[VL]], %[[bdimy:[a-z0-9_]+]] = %[[NW]], %[[bdimz:[a-z0-9_]+]] = %[[bdimz_val]])
func.func @empty_all_known_launch_arg() {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index

  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index

  %par_dim1 = acc.par_width %c2 {par_dim = #acc.par_dim<thread_x>}
  %par_dim2 = acc.par_width %c4 {par_dim = #acc.par_dim<thread_y>}
  %par_dim3 = acc.par_width %c8 {par_dim = #acc.par_dim<thread_z>}
  %par_dim4 = acc.par_width %c16 {par_dim = #acc.par_dim<block_x>}
  %par_dim5 = acc.par_width %c32 {par_dim = #acc.par_dim<block_y>}
  %par_dim6 = acc.par_width %c128 {par_dim = #acc.par_dim<block_z>}
  acc.compute_region launch(%tx = %par_dim1, %ty = %par_dim2, %tz = %par_dim3, %bx = %par_dim4, %by = %par_dim5, %bz = %par_dim6) {
    acc.yield
  } {origin = "acc.parallel"}
  return
}

// CHECK-LABEL: @using_block_args
// 2D block: blockDimX = VL, blockDimY = NW
// CHECK: gpu.launch
// CHECK-SAME: threads({{.*}}) in (%[[bdimx2:[a-z0-9_]+]] = %c2, %[[bdimy2:[a-z0-9_]+]] = %c4, %[[bdimz2:[a-z0-9_]+]] = %c8)
// CHECK: %[[gdimx:.*]] = gpu.grid_dim x
// CHECK: %[[gdimy:.*]] = gpu.grid_dim y
// CHECK: %[[gdimz:.*]] = gpu.grid_dim z
// CHECK: arith.addi %[[gdimx]], %[[gdimy]] : index
// CHECK: arith.subi %[[gdimz]], %c2{{.*}} : index
func.func @using_block_args(%arr : memref<?xf32>) {
  %c2_pw = arith.constant 2 : index
  %c4_pw = arith.constant 4 : index
  %c8_pw = arith.constant 8 : index
  %c16_pw = arith.constant 16 : index
  %c32_pw = arith.constant 32 : index
  %c128_pw = arith.constant 128 : index

  %par_dim1 = acc.par_width %c2_pw {par_dim = #acc.par_dim<thread_x>}
  %par_dim2 = acc.par_width %c4_pw {par_dim = #acc.par_dim<thread_y>}
  %par_dim3 = acc.par_width %c8_pw {par_dim = #acc.par_dim<thread_z>}
  %par_dim4 = acc.par_width %c16_pw {par_dim = #acc.par_dim<block_x>}
  %par_dim5 = acc.par_width %c32_pw {par_dim = #acc.par_dim<block_y>}
  %par_dim6 = acc.par_width %c128_pw {par_dim = #acc.par_dim<block_z>}
  acc.compute_region launch(%tx = %par_dim1, %ty = %par_dim2, %tz = %par_dim3, %bx = %par_dim4, %by = %par_dim5, %bz = %par_dim6) ins(%arg10 = %arr) : (memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 1.000000e+01 : f32
    scf.parallel (%arg8) = (%c0) to (%tx) step (%tx) {
      %22 = arith.muli %ty, %tz : index
      %23 = arith.addi %bx, %by : index
      %24 = arith.subi %bz, %c2 : index
      scf.parallel (%arg9) = (%22) to (%23) step (%24) {
        memref.store %cst, %arg10[%arg9] : memref<?xf32>
        scf.reduce
      }
    }
    acc.yield
  } {origin = "acc.parallel"}
  return
}
