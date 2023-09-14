//--------------------------------------------------------------------------------------------------
// WHEN CREATING A NEW TEST, PLEASE JUST COPY & PASTE WITHOUT EDITS.
//
// Set-up that's shared across all tests in this directory. In principle, this
// config could be moved to lit.local.cfg. However, there are downstream users that
//  do not use these LIT config files. Hence why this is kept inline.
//
// DEFINE: %{sparse_compiler_opts} = enable-runtime-library=true
// DEFINE: %{sparse_compiler_opts_sve} = enable-arm-sve=true %{sparse_compiler_opts}
// DEFINE: %{compile} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts}"
// DEFINE: %{compile_sve} = mlir-opt %s --sparse-compiler="%{sparse_compiler_opts_sve}"
// DEFINE: %{run_libs} = -shared-libs=%mlir_c_runner_utils,%mlir_runner_utils
// DEFINE: %{run_opts} = -e entry -entry-point-result=void
// DEFINE: %{run} = mlir-cpu-runner %{run_opts} %{run_libs}
// DEFINE: %{run_sve} = %mcr_aarch64_cmd --march=aarch64 --mattr="+sve" %{run_opts} %{run_libs}
//
// DEFINE: %{env} =
//--------------------------------------------------------------------------------------------------

// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with direct IR generation and VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#Row = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "dense" ]
}>

#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

#DCSC = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed_nu", "singleton" ]
}>

#SortedCOOPerm = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed_nu", "singleton" ],
  dimToLvl = affine_map<(i,j) -> (j,i)>
}>

#CCCPerm = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed", "compressed", "compressed"],
  dimToLvl = affine_map<(d0, d1, d2) -> (d1, d2, d0)>
}>

module {
  /// uses foreach operator to print coords and values.
  func.func @foreach_print_const() {
    // Initialize a tensor.
    %0 = arith.constant sparse<[[0, 0], [1, 6]], [1.0, 5.0]> : tensor<8x7xf32>
    sparse_tensor.foreach in %0 : tensor<8x7xf32> do {
      ^bb0(%1: index, %2: index, %v: f32) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f32
     }
     return
  }

  /// uses foreach operator to print coords and values.
  func.func @foreach_print_1(%arg0: tensor<2x2xf64, #Row>) {
    sparse_tensor.foreach in %arg0 : tensor<2x2xf64, #Row> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }
     return
  }

  func.func @foreach_print_2(%arg0: tensor<2x2xf64, #CSR>) {
    sparse_tensor.foreach in %arg0 : tensor<2x2xf64, #CSR> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }
     return
  }

  func.func @foreach_print_3(%arg0: tensor<2x2xf64, #DCSC>) {
    sparse_tensor.foreach in %arg0 : tensor<2x2xf64, #DCSC> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }
     return
  }

  func.func @foreach_print_4(%arg0: tensor<2x2xf64, #SortedCOO>) {
    sparse_tensor.foreach in %arg0 : tensor<2x2xf64, #SortedCOO> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }
     return
  }

  func.func @foreach_print_5(%arg0: tensor<2x2xf64, #SortedCOOPerm>) {
    sparse_tensor.foreach in %arg0 : tensor<2x2xf64, #SortedCOOPerm> do {
      ^bb0(%1: index, %2: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %v: f64
     }
     return
  }

  func.func @foreach_print_3d(%arg0: tensor<7x8x9xf64, #CCCPerm>) {
    sparse_tensor.foreach in %arg0 : tensor<7x8x9xf64, #CCCPerm> do {
      ^bb0(%1: index, %2: index, %3: index, %v: f64) :
        vector.print %1: index
        vector.print %2: index
        vector.print %3: index
        vector.print %v: f64
     }
     return
  }


  func.func @foreach_print_dense(%arg0: tensor<2x2xf64>) {
    sparse_tensor.foreach in %arg0 : tensor<2x2xf64> do {
    ^bb0(%1: index, %2: index, %v: f64) :
      vector.print %1: index
      vector.print %2: index
      vector.print %v: f64
   }
   return
  }

  //
  // Main driver.
  //
  func.func @entry() {
    //
    // Initialize a 3-dim dense tensor.
    //
    %src = arith.constant dense<
       [[  1.0,  2.0],
        [  5.0,  6.0]]
    > : tensor<2x2xf64>

    %src3d = arith.constant sparse<
       [[1, 2, 3], [4, 5, 6]], [1.0, 2.0]
    > : tensor<7x8x9xf64>

    //
    // Convert dense tensor directly to various sparse tensors.
    //
    %s1 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #Row>
    %s2 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #CSR>
    %s3 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #DCSC>
    %s4 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #SortedCOO>
    %s5 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #SortedCOOPerm>
    %s6 = sparse_tensor.convert %src3d : tensor<7x8x9xf64>  to tensor<7x8x9xf64, #CCCPerm>
    // CHECK: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    // CHECK-NEXT: 5
    call @foreach_print_const() : () -> ()
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    call @foreach_print_dense(%src) : (tensor<2x2xf64>) -> ()
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    call @foreach_print_1(%s1) : (tensor<2x2xf64, #Row>) -> ()
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    call @foreach_print_2(%s2) : (tensor<2x2xf64, #CSR>) -> ()
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    call @foreach_print_3(%s3) : (tensor<2x2xf64, #DCSC>) -> ()
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    call @foreach_print_4(%s4) : (tensor<2x2xf64, #SortedCOO>) -> ()
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 5
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1
    // CHECK-NEXT: 6
    call @foreach_print_5(%s5) : (tensor<2x2xf64, #SortedCOOPerm>) -> ()

    // CHECK-NEXT: 1
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 1
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 6
    // CHECK-NEXT: 2
    call @foreach_print_3d(%s6): (tensor<7x8x9xf64, #CCCPerm>) -> ()

    bufferization.dealloc_tensor %s1 : tensor<2x2xf64, #Row>
    bufferization.dealloc_tensor %s2 : tensor<2x2xf64, #CSR>
    bufferization.dealloc_tensor %s3 : tensor<2x2xf64, #DCSC>
    bufferization.dealloc_tensor %s4 : tensor<2x2xf64, #SortedCOO>
    bufferization.dealloc_tensor %s5 : tensor<2x2xf64, #SortedCOOPerm>
    bufferization.dealloc_tensor %s6 : tensor<7x8x9xf64, #CCCPerm>

    return
  }
}
