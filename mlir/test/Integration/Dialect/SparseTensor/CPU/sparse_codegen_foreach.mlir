// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#Row = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "dense" ]
}>

#CSR = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed" ]
}>

#DCSC = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed", "compressed" ],
  dimOrdering = affine_map<(i,j) -> (j,i)>
}>

module {

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

    //
    // Convert dense tensor directly to various sparse tensors.
    //
    %s1 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #Row>
    %s2 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #CSR>
    %s3 = sparse_tensor.convert %src : tensor<2x2xf64> to tensor<2x2xf64, #DCSC>
    // CHECK: 0
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
    
    bufferization.dealloc_tensor %s1 : tensor<2x2xf64, #Row>
    bufferization.dealloc_tensor %s2 : tensor<2x2xf64, #CSR>
    bufferization.dealloc_tensor %s3 : tensor<2x2xf64, #DCSC>

    return
  }
}
