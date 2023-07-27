// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler=%{option}
// DEFINE: %{run} = mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_c_runner_utils | \
// DEFINE: FileCheck %s
//
// RUN: %{compile} | %{run}
//

// Do the same run, but now with direct IR generation and, if available, VLA
// vectorization.
// REDEFINE: %{option} = "enable-runtime-library=false vl=4 enable-arm-sve=%ENABLE_VLA"
// REDEFINE: %{run} = %lli_host_or_aarch64_cmd \
// REDEFINE:   --entry-function=entry_lli \
// REDEFINE:   --extra-module=%S/Inputs/main_for_lli.ll \
// REDEFINE:   %VLA_ARCH_ATTR_OPTIONS \
// REDEFINE:   --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// REDEFINE: FileCheck %s
// RUN: %{compile} | mlir-translate -mlir-to-llvmir | %{run}

// TODO: Pack only support CodeGen Path

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>

#SortedCOOI32 = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ],
  posWidth = 32,
  crdWidth = 32
}>

#CSR = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed" ],
  posWidth = 32,
  crdWidth = 32
}>

#BCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "dense", "compressed-hi-nu", "singleton" ]
}>

module {
  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %f0 = arith.constant 0.0 : f64
    %i0 = arith.constant 0 : i32
    //
    // Initialize a 3-dim dense tensor.
    //
    %data = arith.constant dense<
       [  1.0,  2.0,  3.0]
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

    %s4 = sparse_tensor.pack %data, %pos, %index : tensor<3xf64>, tensor<2xindex>, tensor<3x2xindex>
                                          to tensor<10x10xf64, #SortedCOO>
    %s5= sparse_tensor.pack %data, %pos32, %index32 : tensor<3xf64>, tensor<2xi32>, tensor<3x2xi32>
                                           to tensor<10x10xf64, #SortedCOOI32>

    %csr_data = arith.constant dense<
       [  1.0,  2.0,  3.0,  4.0]
    > : tensor<4xf64>

    %csr_pos32 = arith.constant dense<
       [0, 1, 3]
    > : tensor<3xi32>

    %csr_index32 = arith.constant dense<
       [1, 0, 1]
    > : tensor<3xi32>
    %csr= sparse_tensor.pack %csr_data, %csr_pos32, %csr_index32 : tensor<4xf64>, tensor<3xi32>, tensor<3xi32>
                                           to tensor<2x2xf64, #CSR>

    %bdata = arith.constant dense<
       [  1.0,  2.0,  3.0,  4.0,  5.0,  0.0]
    > : tensor<6xf64>

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
    %bs = sparse_tensor.pack %bdata, %bpos, %bindex :
          tensor<6xf64>, tensor<4xindex>,  tensor<6x2xindex> to tensor<2x10x10xf64, #BCOO>

    // CHECK:1
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

    %d_csr = tensor.empty() : tensor<4xf64>
    %p_csr = tensor.empty() : tensor<3xi32>
    %i_csr = tensor.empty() : tensor<3xi32>
    %rd_csr, %rp_csr, %ri_csr, %ld_csr, %lp_csr, %li_csr = sparse_tensor.unpack %csr : tensor<2x2xf64, #CSR>
                 outs(%d_csr, %p_csr, %i_csr : tensor<4xf64>, tensor<3xi32>, tensor<3xi32>)
                 -> tensor<4xf64>, (tensor<3xi32>, tensor<3xi32>), index, (index, index)

    // CHECK-NEXT: ( 1, 2, 3, {{.*}} )
    %vd_csr = vector.transfer_read %rd_csr[%c0], %f0 : tensor<4xf64>, vector<4xf64>
    vector.print %vd_csr : vector<4xf64>

    // CHECK-NEXT:1
    // CHECK-NEXT:2
    // CHECK-NEXT:3
    //
    // CHECK-NEXT:4
    // CHECK-NEXT:5
    //
    // Make sure the trailing zeros are not traversed.
    // CHECK-NOT: 0
    sparse_tensor.foreach in %bs : tensor<2x10x10xf64, #BCOO> do {
      ^bb0(%0: index, %1: index, %2: index, %v: f64) :
        vector.print %v: f64
     }

    %od = tensor.empty() : tensor<3xf64>
    %op = tensor.empty() : tensor<2xi32>
    %oi = tensor.empty() : tensor<3x2xi32>
    %d, %p, %i, %dl, %pl, %il = sparse_tensor.unpack %s5 : tensor<10x10xf64, #SortedCOOI32>
                 outs(%od, %op, %oi : tensor<3xf64>, tensor<2xi32>, tensor<3x2xi32>)
                 -> tensor<3xf64>, (tensor<2xi32>, tensor<3x2xi32>), index, (index, index)

    // CHECK-NEXT: ( 1, 2, 3 )
    %vd = vector.transfer_read %d[%c0], %f0 : tensor<3xf64>, vector<3xf64>
    vector.print %vd : vector<3xf64>

    // CHECK-NEXT: ( ( 1, 2 ), ( 5, 6 ), ( 7, 8 ) )
    %vi = vector.transfer_read %i[%c0, %c0], %i0 : tensor<3x2xi32>, vector<3x2xi32>
    vector.print %vi : vector<3x2xi32>


    %bod = tensor.empty() : tensor<6xf64>
    %bop = tensor.empty() : tensor<4xindex>
    %boi = tensor.empty() : tensor<6x2xindex>
    %bd, %bp, %bi, %ld, %lp, %li = sparse_tensor.unpack %bs : tensor<2x10x10xf64, #BCOO>
                    outs(%bod, %bop, %boi : tensor<6xf64>, tensor<4xindex>, tensor<6x2xindex>)
                    -> tensor<6xf64>, (tensor<4xindex>, tensor<6x2xindex>), index, (index, index)

    // CHECK-NEXT: ( 1, 2, 3, 4, 5, {{.*}} )
    %vbd = vector.transfer_read %bd[%c0], %f0 : tensor<6xf64>, vector<6xf64>
    vector.print %vbd : vector<6xf64>
    // CHECK-NEXT: 5
    vector.print %ld : index

    // CHECK-NEXT: ( ( 1, 2 ), ( 5, 6 ), ( 7, 8 ), ( 2, 3 ), ( 4, 2 ), ( {{.*}}, {{.*}} ) )
    %vbi = vector.transfer_read %bi[%c0, %c0], %c0 : tensor<6x2xindex>, vector<6x2xindex>
    vector.print %vbi : vector<6x2xindex>
    // CHECK-NEXT: 10
    vector.print %li : index

    return
  }
}
