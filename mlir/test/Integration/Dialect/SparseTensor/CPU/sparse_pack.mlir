// DEFINE: %{option} = enable-runtime-library=false
// DEFINE: %{command} = mlir-opt %s --sparse-compiler=%{option} | \
// DEFINE: mlir-cpu-runner \
// DEFINE:  -e entry -entry-point-result=void  \
// DEFINE:  -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext | \
// DEFINE: FileCheck %s
//
// RUN: %{command}
//

// TODO: Pack only support CodeGen Path

#SortedCOO = #sparse_tensor.encoding<{
  dimLevelType = [ "compressed-nu", "singleton" ]
}>

module {
  //
  // Main driver.
  //
  func.func @entry() {
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

    %s4 = sparse_tensor.pack %data, %index : tensor<3xf64>, tensor<3x2xindex>
                                          to tensor<10x10xf64, #SortedCOO>
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
    return
  }
}
