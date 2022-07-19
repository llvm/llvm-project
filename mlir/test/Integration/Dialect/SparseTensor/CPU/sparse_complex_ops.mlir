// RUN: mlir-opt %s --sparse-compiler | \
// RUN: mlir-cpu-runner \
// RUN:  -e entry -entry-point-result=void  \
// RUN:  -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

#SparseVector = #sparse_tensor.encoding<{dimLevelType = ["compressed"]}>

#trait_op1 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = OP a(i)"
}

#trait_op2 = {
  indexing_maps = [
    affine_map<(i) -> (i)>,  // a (in)
    affine_map<(i) -> (i)>,  // b (in)
    affine_map<(i) -> (i)>   // x (out)
  ],
  iterator_types = ["parallel"],
  doc = "x(i) = a(i) OP b(i)"
}

module {
  func.func @cops(%arga: tensor<?xcomplex<f64>, #SparseVector>,
                  %argb: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op2
       ins(%arga, %argb: tensor<?xcomplex<f64>, #SparseVector>,
                         tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %b: complex<f64>, %x: complex<f64>):
          %1 = complex.neg %b : complex<f64>
          %2 = complex.sub %a, %1 : complex<f64>
          linalg.yield %2 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @csin(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.sin %a : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @complex_sqrt(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.sqrt %a : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @complex_tanh(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
       ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.tanh %a : complex<f64>
          linalg.yield %1 : complex<f64>
   } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @clog1p_expm1(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.log1p %a : complex<f64>
          %2 = complex.expm1 %1 : complex<f64>
          linalg.yield %2 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @cdiv(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xcomplex<f64>, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xcomplex<f64>, #SparseVector>
    %c = complex.constant [2.0 : f64, 0.0 : f64] : complex<f64>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xcomplex<f64>, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: complex<f64>):
          %1 = complex.div %a, %c  : complex<f64>
          linalg.yield %1 : complex<f64>
    } -> tensor<?xcomplex<f64>, #SparseVector>
    return %0 : tensor<?xcomplex<f64>, #SparseVector>
  }

  func.func @cabs(%arga: tensor<?xcomplex<f64>, #SparseVector>)
                 -> tensor<?xf64, #SparseVector> {
    %c0 = arith.constant 0 : index
    %d = tensor.dim %arga, %c0 : tensor<?xcomplex<f64>, #SparseVector>
    %xv = bufferization.alloc_tensor(%d) : tensor<?xf64, #SparseVector>
    %0 = linalg.generic #trait_op1
       ins(%arga: tensor<?xcomplex<f64>, #SparseVector>)
        outs(%xv: tensor<?xf64, #SparseVector>) {
        ^bb(%a: complex<f64>, %x: f64):
          %1 = complex.abs %a : complex<f64>
          linalg.yield %1 : f64
    } -> tensor<?xf64, #SparseVector>
    return %0 : tensor<?xf64, #SparseVector>
  }

  func.func @dumpc(%arg0: tensor<?xcomplex<f64>, #SparseVector>, %d: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %mem = sparse_tensor.values %arg0 : tensor<?xcomplex<f64>, #SparseVector> to memref<?xcomplex<f64>>
    scf.for %i = %c0 to %d step %c1 {
       %v = memref.load %mem[%i] : memref<?xcomplex<f64>>
       %real = complex.re %v : complex<f64>
       %imag = complex.im %v : complex<f64>
       vector.print %real : f64
       vector.print %imag : f64
    }
    return
  }

  func.func @dumpf(%arg0: tensor<?xf64, #SparseVector>) {
    %c0 = arith.constant 0 : index
    %d0 = arith.constant 0.0 : f64
    %values = sparse_tensor.values %arg0 : tensor<?xf64, #SparseVector> to memref<?xf64>
    %0 = vector.transfer_read %values[%c0], %d0: memref<?xf64>, vector<3xf64>
    vector.print %0 : vector<3xf64>
    return
  }

  // Driver method to call and verify complex kernels.
  func.func @entry() {
    // Setup sparse vectors.
    %v1 = arith.constant sparse<
       [ [0], [28], [31] ],
         [ (-5.13, 2.0), (3.0, 4.0), (5.0, 6.0) ] > : tensor<32xcomplex<f64>>
    %v2 = arith.constant sparse<
       [ [1], [28], [31] ],
         [ (1.0, 0.0), (-2.0, 0.0), (3.0, 0.0) ] > : tensor<32xcomplex<f64>>
    %sv1 = sparse_tensor.convert %v1 : tensor<32xcomplex<f64>> to tensor<?xcomplex<f64>, #SparseVector>
    %sv2 = sparse_tensor.convert %v2 : tensor<32xcomplex<f64>> to tensor<?xcomplex<f64>, #SparseVector>

    // Call sparse vector kernels.
    %0 = call @cops(%sv1, %sv2)
       : (tensor<?xcomplex<f64>, #SparseVector>,
          tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %1 = call @csin(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %2 = call @complex_sqrt(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %3 = call @complex_tanh(%sv2)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %4 = call @clog1p_expm1(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %5 = call @cdiv(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xcomplex<f64>, #SparseVector>
    %6 = call @cabs(%sv1)
       : (tensor<?xcomplex<f64>, #SparseVector>) -> tensor<?xf64, #SparseVector>

    //
    // Verify the results.
    //
    %d3 = arith.constant 3 : index
    %d4 = arith.constant 4 : index
    // CHECK: -5.13
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 0
    // CHECK-NEXT: 1
    // CHECK-NEXT: 4
    // CHECK-NEXT: 8
    // CHECK-NEXT: 6
    call @dumpc(%0, %d4) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    // CHECK-NEXT: 3.43887
    // CHECK-NEXT: 1.47097
    // CHECK-NEXT: 3.85374
    // CHECK-NEXT: -27.0168
    // CHECK-NEXT: -193.43
    // CHECK-NEXT: 57.2184
    call @dumpc(%1, %d3) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    // CHECK-NEXT: 0.433635
    // CHECK-NEXT: 2.30609
    // CHECK-NEXT: 2
    // CHECK-NEXT: 1
    // CHECK-NEXT: 2.53083
    // CHECK-NEXT: 1.18538
    call @dumpc(%2, %d3) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    // CHECK-NEXT: 0.761594
    // CHECK-NEXT: 0
    // CHECK-NEXT: -0.964028
    // CHECK-NEXT: 0
    // CHECK-NEXT: 0.995055
    // CHECK-NEXT: 0
    call @dumpc(%3, %d3) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    // CHECK-NEXT: -5.13
    // CHECK-NEXT: 2
    // CHECK-NEXT: 3
    // CHECK-NEXT: 4
    // CHECK-NEXT: 5
    // CHECK-NEXT: 6
    call @dumpc(%4, %d3) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    // CHECK-NEXT: -2.565
    // CHECK-NEXT: 1
    // CHECK-NEXT: 1.5
    // CHECK-NEXT: 2
    // CHECK-NEXT: 2.5
    // CHECK-NEXT: 3
    call @dumpc(%5, %d3) : (tensor<?xcomplex<f64>, #SparseVector>, index) -> ()
    // CHECK-NEXT: ( 5.50608, 5, 7.81025 )
    call @dumpf(%6) : (tensor<?xf64, #SparseVector>) -> ()

    // Release the resources.
    bufferization.dealloc_tensor %sv1 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %sv2 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %0 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %1 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %2 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %3 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %4 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %5 : tensor<?xcomplex<f64>, #SparseVector>
    bufferization.dealloc_tensor %6 : tensor<?xf64, #SparseVector>
    return
  }
}
