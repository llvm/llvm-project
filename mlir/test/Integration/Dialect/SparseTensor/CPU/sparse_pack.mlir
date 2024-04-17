//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
// do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparsifier_opts} = enable-runtime-library=true
// DEFINE: %{sparsifier_opts_sve} = enable-arm-sve=true %{sparsifier_opts}
// DEFINE: %{compile} = mlir-opt %s --sparsifier="%{sparsifier_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparsifier="%{sparsifier_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e main -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparsifier_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#SortedCOO = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton)
}>

#SortedCOOI32 = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : compressed(nonunique), d1 : singleton),
  posWidth = 32,
  crdWidth = 32
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed),
  posWidth = 32,
  crdWidth = 32
}>

#BCOO = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : loose_compressed(nonunique), d2 : singleton)
}>

module {
  //
  // Main driver.
  //
  func.func @main() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f64
    %i0 = arith.constant 0 : i32

    //
    // Setup COO.
    //

    %data = arith.constant dense<
       [ 1.0,  2.0,  3.0 ]
    > : tensor<3xf64>

    %pos = arith.constant dense<
       [0, 3]
    > : tensor<2xindex>

    %index = arith.constant dense<
       [[  1,  2],
        [  5,  6],
        [  7,  8]]
    > : tensor<3x2xindex>

    %pos32 = arith.constant dense<
       [0, 3]
    > : tensor<2xi32>

    %index32 = arith.constant dense<
       [[  1,  2],
        [  5,  6],
        [  7,  8]]
    > : tensor<3x2xi32>

    %s4 = sparse_tensor.assemble (%pos, %index), %data : (tensor<2xindex>, tensor<3x2xindex>), tensor<3xf64>
                                          to tensor<10x10xf64, #SortedCOO>
    %s5 = sparse_tensor.assemble (%pos32, %index32), %data : (tensor<2xi32>, tensor<3x2xi32>), tensor<3xf64>
                                          to tensor<10x10xf64, #SortedCOOI32>

    //
    // Setup CSR.
    //

    %csr_data = arith.constant dense<
       [ 1.0,  2.0,  3.0 ]
    > : tensor<3xf64>

    %csr_pos32 = arith.constant dense<
       [0, 1, 3]
    > : tensor<3xi32>

    %csr_index32 = arith.constant dense<
       [1, 0, 1]
    > : tensor<3xi32>
    %csr = sparse_tensor.assemble (%csr_pos32, %csr_index32), %csr_data : (tensor<3xi32>, tensor<3xi32>), tensor<3xf64>
                                           to tensor<2x2xf64, #CSR>

    //
    // Setup BCOO.
    //

    %bdata = arith.constant dense<
       [ 1.0,  2.0,  3.0,  4.0,  5.0 ]
    > : tensor<5xf64>

    %bpos = arith.constant dense<
       [0, 3, 3, 5]
    > : tensor<4xindex>

    %bindex = arith.constant dense<
      [[  1,  2],
       [  5,  6],
       [  7,  8],
       [  2,  3],
       [  4,  2],
       [ 10, 10]]
    > : tensor<6x2xindex>

    %bs = sparse_tensor.assemble (%bpos, %bindex), %bdata :
          (tensor<4xindex>,  tensor<6x2xindex>), tensor<5xf64> to tensor<2x10x10xf64, #BCOO>

    //
    // Verify results.
    //

    // CHECK:     1
    // CHECK-NEXT:2
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:5
    // CHECK-NEXT:6
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:7
    // CHECK-NEXT:8
    // CHECK-NEXT:3
    sparse_tensor.foreach in %s4 : tensor<10x10xf64, #SortedCOO> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
    }

    // CHECK-NEXT:1
    // CHECK-NEXT:2
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:5
    // CHECK-NEXT:6
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:7
    // CHECK-NEXT:8
    // CHECK-NEXT:3
    sparse_tensor.foreach in %s5 : tensor<10x10xf64, #SortedCOOI32> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
    }

    // CHECK-NEXT:0
    // CHECK-NEXT:1
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:1
    // CHECK-NEXT:0
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:1
    // CHECK-NEXT:1
    // CHECK-NEXT:3
    sparse_tensor.foreach in %csr : tensor<2x2xf64, #CSR> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
    }

    // CHECK-NEXT:0
    // CHECK-NEXT:1
    // CHECK-NEXT:2
    // CHECK-NEXT:1
    //
    // CHECK-NEXT:0
    // CHECK-NEXT:5
    // CHECK-NEXT:6
    // CHECK-NEXT:2
    //
    // CHECK-NEXT:0
    // CHECK-NEXT:7
    // CHECK-NEXT:8
    // CHECK-NEXT:3
    //
    // CHECK-NEXT:1
    // CHECK-NEXT:2
    // CHECK-NEXT:3
    // CHECK-NEXT:4
    //
    // CHECK-NEXT:1
    // CHECK-NEXT:4
    // CHECK-NEXT:2
    // CHECK-NEXT:5
    sparse_tensor.foreach in %bs : tensor<2x10x10xf64, #BCOO> do {
      ^bb0(%0: index, %1: index, %2: index, %v: f64) :
        vector.print %0: index
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
    }

    //
    // Verify disassemble operations.
    //

    %od = tensor.empty() : tensor<3xf64>
    %op = tensor.empty() : tensor<2xi32>
    %oi = tensor.empty() : tensor<3x2xi32>
    %p, %i, %d, %pl, %il, %dl = sparse_tensor.disassemble %s5 : tensor<10x10xf64, #SortedCOOI32>
                 out_lvls(%op, %oi : tensor<2xi32>, tensor<3x2xi32>)
                 out_vals(%od : tensor<3xf64>)
                 -> (tensor<2xi32>, tensor<3x2xi32>), tensor<3xf64>, (i32, i64), index

    // CHECK-NEXT: ( 1, 2, 3 )
    %vd = vector.transfer_read %d[%c0], %f0 : tensor<3xf64>, vector<3xf64>
    vector.print %vd : vector<3xf64>

    // CHECK-NEXT: ( ( 1, 2 ), ( 5, 6 ), ( 7, 8 ) )
    %vi = vector.transfer_read %i[%c0, %c0], %i0 : tensor<3x2xi32>, vector<3x2xi32>
    vector.print %vi : vector<3x2xi32>

    // CHECK-NEXT: 3
    vector.print %dl : index

    %d_csr = tensor.empty() : tensor<4xf64>
    %p_csr = tensor.empty() : tensor<3xi32>
    %i_csr = tensor.empty() : tensor<3xi32>
    %rp_csr, %ri_csr, %rd_csr, %lp_csr, %li_csr, %ld_csr = sparse_tensor.disassemble %csr : tensor<2x2xf64, #CSR>
                 out_lvls(%p_csr, %i_csr : tensor<3xi32>, tensor<3xi32>)
                 out_vals(%d_csr : tensor<4xf64>)
                 -> (tensor<3xi32>, tensor<3xi32>), tensor<4xf64>, (i32, i64), index

    // CHECK-NEXT: ( 1, 2, 3 )
    %vd_csr = vector.transfer_read %rd_csr[%c0], %f0 : tensor<4xf64>, vector<3xf64>
    vector.print %vd_csr : vector<3xf64>

    // CHECK-NEXT: 3
    vector.print %ld_csr : index

    %bod = tensor.empty() : tensor<6xf64>
    %bop = tensor.empty() : tensor<4xindex>
    %boi = tensor.empty() : tensor<6x2xindex>
    %bp, %bi, %bd, %lp, %li, %ld = sparse_tensor.disassemble %bs : tensor<2x10x10xf64, #BCOO>
                    out_lvls(%bop, %boi : tensor<4xindex>, tensor<6x2xindex>)
                    out_vals(%bod : tensor<6xf64>)
                    -> (tensor<4xindex>, tensor<6x2xindex>), tensor<6xf64>, (i32, tensor<i64>), index

    // CHECK-NEXT: ( 1, 2, 3, 4, 5 )
    %vbd = vector.transfer_read %bd[%c0], %f0 : tensor<6xf64>, vector<5xf64>
    vector.print %vbd : vector<5xf64>

    // CHECK-NEXT: 5
    vector.print %ld : index

    // CHECK-NEXT: ( ( 1, 2 ), ( 5, 6 ), ( 7, 8 ), ( 2, 3 ), ( 4, 2 ), ( {{.*}}, {{.*}} ) )
    %vbi = vector.transfer_read %bi[%c0, %c0], %c0 : tensor<6x2xindex>, vector<6x2xindex>
    vector.print %vbi : vector<6x2xindex>

    // CHECK-NEXT: 10
    %si = tensor.extract %li[] : tensor<i64>
    vector.print %si : i64

    // TODO: This check is no longer needed once the codegen path uses the
    // buffer deallocation pass. "dealloc_tensor" turn into a no-op in the
    // codegen path.
    %has_runtime = sparse_tensor.has_runtime_library
    scf.if %has_runtime {
      // sparse_tensor.assemble copies buffers when running with the runtime
      // library. Deallocations are not needed when running in codegen mode.
      bufferization.dealloc_tensor %s4 : tensor<10x10xf64, #SortedCOO>
      bufferization.dealloc_tensor %s5 : tensor<10x10xf64, #SortedCOOI32>
      bufferization.dealloc_tensor %csr : tensor<2x2xf64, #CSR>
      bufferization.dealloc_tensor %bs : tensor<2x10x10xf64, #BCOO>
    }

    bufferization.dealloc_tensor %li : tensor<i64>
    bufferization.dealloc_tensor %od : tensor<3xf64>
    bufferization.dealloc_tensor %op : tensor<2xi32>
    bufferization.dealloc_tensor %oi : tensor<3x2xi32>
    bufferization.dealloc_tensor %d_csr : tensor<4xf64>
    bufferization.dealloc_tensor %p_csr : tensor<3xi32>
    bufferization.dealloc_tensor %i_csr : tensor<3xi32>
    bufferization.dealloc_tensor %bod : tensor<6xf64>
    bufferization.dealloc_tensor %bop : tensor<4xindex>
    bufferization.dealloc_tensor %boi : tensor<6x2xindex>

    return
  }
}
