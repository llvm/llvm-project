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
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

#SortedCOOI32 = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ],
  posWidth = 32,
  crdWidth = 32
}>

#BCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "dense", "compressed-hi-nu", "singleton" ]
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

    %index = arith.constant dense<
       [[  1,  2],
        [  5,  6],
        [  7,  8]]
    > : tensor<3x2xindex>

    %index32 = arith.constant dense<
       [[  1,  2],
        [  5,  6],
        [  7,  8]]
    > : tensor<3x2xi32>

    %s4 = sparse_tensor.pack %data, %index : tensor<3xf64>, tensor<3x2xindex>
                                          to tensor<10x10xf64, #SortedCOO>
    %s5= sparse_tensor.pack %data, %index32 : tensor<3xf64>, tensor<3x2xi32>
                                           to tensor<10x10xf64, #SortedCOOI32>

    %bdata = arith.constant dense<
       [[  1.0,  2.0,  3.0],
        [  4.0,  5.0,  0.0]]
    > : tensor<2x3xf64>

    %bindex = arith.constant dense<
      [[[  1,  2],
        [  5,  6],
        [  7,  8]],
       [[  2,  3],
        [  4,  2],
        [ 10, 10]]]
    > : tensor<2x3x2xindex>
    %bs = sparse_tensor.pack %bdata, %bindex batched_lvls = 1 :
          tensor<2x3xf64>, tensor<2x3x2xindex> to tensor<2x10x10xf64, #BCOO>

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

    %d, %i, %n = sparse_tensor.unpack %s5 : tensor<10x10xf64, #SortedCOOI32>
                                         to tensor<3xf64>, tensor<3x2xi32>, i32

    // CHECK-NEXT: ( 1, 2, 3 )
    %vd = vector.transfer_read %d[%c0], %f0 : tensor<3xf64>, vector<3xf64>
    vector.print %vd : vector<3xf64>

    // CHECK-NEXT: ( ( 1, 2 ), ( 5, 6 ), ( 7, 8 ) )
    %vi = vector.transfer_read %i[%c0, %c0], %i0 : tensor<3x2xi32>, vector<3x2xi32>
    vector.print %vi : vector<3x2xi32>

    // CHECK-NEXT: 3
    vector.print %n : i32


    %bd, %bi, %bn = sparse_tensor.unpack %bs batched_lvls=1 :
       tensor<2x10x10xf64, #BCOO> to tensor<2x3xf64>, tensor<2x3x2xindex>, i32

    // CHECK-NEXT: ( ( 1, 2, 3 ), ( 4, 5, 0 ) )
    %vbd = vector.transfer_read %bd[%c0, %c0], %f0 : tensor<2x3xf64>, vector<2x3xf64>
    vector.print %vbd : vector<2x3xf64>

    // CHECK-NEXT: ( ( ( 1, 2 ), ( 5, 6 ), ( 7, 8 ) ), ( ( 2, 3 ), ( 4, 2 ), ( 0, 0 ) ) )
    %vbi = vector.transfer_read %bi[%c0, %c0, %c0], %c0 : tensor<2x3x2xindex>, vector<2x3x2xindex>
    vector.print %vbi : vector<2x3x2xindex>

    // CHECK-NEXT: 3
    vector.print %bn : i32

    %d1, %i1, %n1 = sparse_tensor.unpack %s4 : tensor<10x10xf64, #SortedCOO>
                                         to tensor<3xf64>, tensor<3x2xindex>, index
    // FIXME: This should be freed by one-shot-bufferization.
    bufferization.dealloc_tensor %bd : tensor<2x3xf64>
    bufferization.dealloc_tensor %bi : tensor<2x3x2xindex>
    return
  }
}
