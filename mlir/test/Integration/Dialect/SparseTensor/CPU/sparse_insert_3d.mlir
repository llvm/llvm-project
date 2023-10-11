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

// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with vectorization.
// REDEFINE: %{sparse_compiler_opts} = enable-runtime-library=false vl=2 reassociate-fp-reductions=true enable-index-optimizations=true
// RUN: %{compile} | %{run} | FileCheck %s
//
// Do the same run, but now with  VLA vectorization.
// RUN: %if mlir_arm_sve_tests %{ %{compile_sve} | %{run_sve} | FileCheck %s %}

#TensorCSR = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : dense, d2 : compressed)
}>

#TensorRow = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed, d2 : dense)
}>

#CCoo = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : compressed, d1 : compressed(nonunique), d2 : singleton)
}>

#DCoo = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed(nonunique), d2 : singleton)
}>


module {

  func.func @dump(%arg0: tensor<5x4x3xf64, #TensorCSR>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %i0 = sparse_tensor.coordinates  %arg0 { level = 0 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %p2 = sparse_tensor.positions %arg0 { level = 2 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %i2 = sparse_tensor.coordinates  %arg0 { level = 2 : index } : tensor<5x4x3xf64, #TensorCSR> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<5x4x3xf64, #TensorCSR> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vi0 : vector<2xindex>
    %vp2 = vector.transfer_read %p2[%c0], %c0: memref<?xindex>, vector<9xindex>
    vector.print %vp2 : vector<9xindex>
    %vi2 = vector.transfer_read %i2[%c0], %c0: memref<?xindex>, vector<5xindex>
    vector.print %vi2 : vector<5xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<5xf64>
    vector.print %vv : vector<5xf64>
    return
  }

  func.func @dump_row(%arg0: tensor<5x4x3xf64, #TensorRow>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %i0 = sparse_tensor.coordinates  %arg0 { level = 0 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %p1 = sparse_tensor.positions %arg0 { level = 1 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %i1 = sparse_tensor.coordinates  %arg0 { level = 1 : index } : tensor<5x4x3xf64, #TensorRow> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<5x4x3xf64, #TensorRow> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vi0 : vector<2xindex>
    %vp1 = vector.transfer_read %p1[%c0], %c0: memref<?xindex>, vector<3xindex>
    vector.print %vp1 : vector<3xindex>
    %vi1 = vector.transfer_read %i1[%c0], %c0: memref<?xindex>, vector<4xindex>
    vector.print %vi1 : vector<4xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<12xf64>
    vector.print %vv : vector<12xf64>
    return
  }

func.func @dump_ccoo(%arg0: tensor<5x4x3xf64, #CCoo>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %p0 = sparse_tensor.positions %arg0 { level = 0 : index } : tensor<5x4x3xf64, #CCoo> to memref<?xindex>
    %i0 = sparse_tensor.coordinates  %arg0 { level = 0 : index } : tensor<5x4x3xf64, #CCoo> to memref<?xindex>
    %p1 = sparse_tensor.positions %arg0 { level = 1 : index } : tensor<5x4x3xf64, #CCoo> to memref<?xindex>
    %i1 = sparse_tensor.coordinates  %arg0 { level = 1 : index } : tensor<5x4x3xf64, #CCoo> to memref<?xindex>
    %i2 = sparse_tensor.coordinates  %arg0 { level = 2 : index } : tensor<5x4x3xf64, #CCoo> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<5x4x3xf64, #CCoo> to memref<?xf64>
    %vp0 = vector.transfer_read %p0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vp0 : vector<2xindex>
    %vi0 = vector.transfer_read %i0[%c0], %c0: memref<?xindex>, vector<2xindex>
    vector.print %vi0 : vector<2xindex>
    %vp1 = vector.transfer_read %p1[%c0], %c0: memref<?xindex>, vector<3xindex>
    vector.print %vp1 : vector<3xindex>
    %vi1 = vector.transfer_read %i1[%c0], %c0: memref<?xindex>, vector<5xindex>
    vector.print %vi1 : vector<5xindex>
    %vi2 = vector.transfer_read %i2[%c0], %c0: memref<?xindex>, vector<5xindex>
    vector.print %vi2 : vector<5xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<5xf64>
    vector.print %vv : vector<5xf64>
    return
  }

func.func @dump_dcoo(%arg0: tensor<5x4x3xf64, #DCoo>) {
    %c0 = arith.constant 0 : index
    %fu = arith.constant 99.0 : f64
    %p1 = sparse_tensor.positions %arg0 { level = 1 : index } : tensor<5x4x3xf64, #DCoo> to memref<?xindex>
    %i1 = sparse_tensor.coordinates  %arg0 { level = 1 : index } : tensor<5x4x3xf64, #DCoo> to memref<?xindex>
    %i2 = sparse_tensor.coordinates  %arg0 { level = 2 : index } : tensor<5x4x3xf64, #DCoo> to memref<?xindex>
    %v = sparse_tensor.values %arg0 : tensor<5x4x3xf64, #DCoo> to memref<?xf64>
    %vp1 = vector.transfer_read %p1[%c0], %c0: memref<?xindex>, vector<6xindex>
    vector.print %vp1 : vector<6xindex>
    %vi1 = vector.transfer_read %i1[%c0], %c0: memref<?xindex>, vector<5xindex>
    vector.print %vi1 : vector<5xindex>
    %vi2 = vector.transfer_read %i2[%c0], %c0: memref<?xindex>, vector<5xindex>
    vector.print %vi2 : vector<5xindex>
    %vv = vector.transfer_read %v[%c0], %fu: memref<?xf64>, vector<5xf64>
    vector.print %vv : vector<5xf64>
    return
}

  //
  // Main driver.
  //
  func.func @entry() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %f1 = arith.constant 1.1 : f64
    %f2 = arith.constant 2.2 : f64
    %f3 = arith.constant 3.3 : f64
    %f4 = arith.constant 4.4 : f64
    %f5 = arith.constant 5.5 : f64

    //
    // CHECK:      ( 0, 2 )
    // CHECK-NEXT: ( 3, 4 )
    // CHECK-NEXT: ( 0, 2, 2, 2, 3, 3, 3, 4, 5 )
    // CHECK-NEXT: ( 1, 2, 1, 2, 2 )
    // CHECK-NEXT: ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    //
    %tensora = tensor.empty() : tensor<5x4x3xf64, #TensorCSR>
    %tensor1 = sparse_tensor.insert %f1 into %tensora[%c3, %c0, %c1] : tensor<5x4x3xf64, #TensorCSR>
    %tensor2 = sparse_tensor.insert %f2 into %tensor1[%c3, %c0, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensor3 = sparse_tensor.insert %f3 into %tensor2[%c3, %c3, %c1] : tensor<5x4x3xf64, #TensorCSR>
    %tensor4 = sparse_tensor.insert %f4 into %tensor3[%c4, %c2, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensor5 = sparse_tensor.insert %f5 into %tensor4[%c4, %c3, %c2] : tensor<5x4x3xf64, #TensorCSR>
    %tensorm = sparse_tensor.load %tensor5 hasInserts : tensor<5x4x3xf64, #TensorCSR>
    call @dump(%tensorm) : (tensor<5x4x3xf64, #TensorCSR>) -> ()

    //
    // CHECK-NEXT: ( 0, 2 )
    // CHECK-NEXT: ( 3, 4 )
    // CHECK-NEXT: ( 0, 2, 4 )
    // CHECK-NEXT: ( 0, 3, 2, 3 )
    // CHECK-NEXT: ( 0, 1.1, 2.2, 0, 3.3, 0, 0, 0, 4.4, 0, 0, 5.5 )
    //
    %rowa = tensor.empty() : tensor<5x4x3xf64, #TensorRow>
    %row1 = sparse_tensor.insert %f1 into %rowa[%c3, %c0, %c1] : tensor<5x4x3xf64, #TensorRow>
    %row2 = sparse_tensor.insert %f2 into %row1[%c3, %c0, %c2] : tensor<5x4x3xf64, #TensorRow>
    %row3 = sparse_tensor.insert %f3 into %row2[%c3, %c3, %c1] : tensor<5x4x3xf64, #TensorRow>
    %row4 = sparse_tensor.insert %f4 into %row3[%c4, %c2, %c2] : tensor<5x4x3xf64, #TensorRow>
    %row5 = sparse_tensor.insert %f5 into %row4[%c4, %c3, %c2] : tensor<5x4x3xf64, #TensorRow>
    %rowm = sparse_tensor.load %row5 hasInserts : tensor<5x4x3xf64, #TensorRow>
    call @dump_row(%rowm) : (tensor<5x4x3xf64, #TensorRow>) -> ()

    //
    // CHECK: ( 0, 2 )
    // CHECK-NEXT: ( 3, 4 )
    // CHECK-NEXT: ( 0, 3, 5 )
    // CHECK-NEXT: ( 0, 0, 3, 2, 3 )
    // CHECK-NEXT: ( 1, 2, 1, 2, 2 )
    // CHECK-NEXT: ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    %ccoo = tensor.empty() : tensor<5x4x3xf64, #CCoo>
    %ccoo1 = sparse_tensor.insert %f1 into %ccoo[%c3, %c0, %c1] : tensor<5x4x3xf64, #CCoo>
    %ccoo2 = sparse_tensor.insert %f2 into %ccoo1[%c3, %c0, %c2] : tensor<5x4x3xf64, #CCoo>
    %ccoo3 = sparse_tensor.insert %f3 into %ccoo2[%c3, %c3, %c1] : tensor<5x4x3xf64, #CCoo>
    %ccoo4 = sparse_tensor.insert %f4 into %ccoo3[%c4, %c2, %c2] : tensor<5x4x3xf64, #CCoo>
    %ccoo5 = sparse_tensor.insert %f5 into %ccoo4[%c4, %c3, %c2] : tensor<5x4x3xf64, #CCoo>
    %ccoom = sparse_tensor.load %ccoo5 hasInserts : tensor<5x4x3xf64, #CCoo>
    call @dump_ccoo(%ccoom) : (tensor<5x4x3xf64, #CCoo>) -> ()

    //
    // CHECK-NEXT: ( 0, 0, 0, 0, 3, 5 )
    // CHECK-NEXT: ( 0, 0, 3, 2, 3 )
    // CHECK-NEXT: ( 1, 2, 1, 2, 2 )
    // CHECK-NEXT: ( 1.1, 2.2, 3.3, 4.4, 5.5 )
    %dcoo = tensor.empty() : tensor<5x4x3xf64, #DCoo>
    %dcoo1 = sparse_tensor.insert %f1 into %dcoo[%c3, %c0, %c1] : tensor<5x4x3xf64, #DCoo>
    %dcoo2 = sparse_tensor.insert %f2 into %dcoo1[%c3, %c0, %c2] : tensor<5x4x3xf64, #DCoo>
    %dcoo3 = sparse_tensor.insert %f3 into %dcoo2[%c3, %c3, %c1] : tensor<5x4x3xf64, #DCoo>
    %dcoo4 = sparse_tensor.insert %f4 into %dcoo3[%c4, %c2, %c2] : tensor<5x4x3xf64, #DCoo>
    %dcoo5 = sparse_tensor.insert %f5 into %dcoo4[%c4, %c3, %c2] : tensor<5x4x3xf64, #DCoo>
    %dcoom = sparse_tensor.load %dcoo5 hasInserts : tensor<5x4x3xf64, #DCoo>
    call @dump_dcoo(%dcoom) : (tensor<5x4x3xf64, #DCoo>) -> ()

    // NOE sanity check.
    //
    // CHECK-NEXT: 5
    // CHECK-NEXT: 12
    // CHECK-NEXT: 5
    // CHECK-NEXT: 5
    //
    %noe1 = sparse_tensor.number_of_entries %tensorm : tensor<5x4x3xf64, #TensorCSR>
    vector.print %noe1 : index
    %noe2 = sparse_tensor.number_of_entries %rowm : tensor<5x4x3xf64, #TensorRow>
    vector.print %noe2 : index
    %noe3 = sparse_tensor.number_of_entries %ccoom : tensor<5x4x3xf64, #CCoo>
    vector.print %noe3 : index
    %noe4 = sparse_tensor.number_of_entries %dcoom : tensor<5x4x3xf64, #DCoo>
    vector.print %noe4 : index

    // Release resources.
    bufferization.dealloc_tensor %tensorm : tensor<5x4x3xf64, #TensorCSR>
    bufferization.dealloc_tensor %rowm : tensor<5x4x3xf64, #TensorRow>
    bufferization.dealloc_tensor %ccoom : tensor<5x4x3xf64, #CCoo>
    bufferization.dealloc_tensor %dcoom : tensor<5x4x3xf64, #DCoo>

    return
  }
}
