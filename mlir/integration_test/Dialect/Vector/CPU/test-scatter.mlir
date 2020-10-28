// RUN: mlir-opt %s -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @scatter8(%base: memref<?xf32>,
               %indices: vector<8xi32>,
               %mask: vector<8xi1>, %value: vector<8xf32>) {
  vector.scatter %base, %indices, %mask, %value
    : vector<8xi32>, vector<8xi1>, vector<8xf32> into memref<?xf32>
  return
}

func @printmem(%A: memref<?xf32>) {
  %f = constant 0.0: f32
  %0 = vector.broadcast %f : f32 to vector<8xf32>
  %1 = constant 0: index
  %2 = load %A[%1] : memref<?xf32>
  %3 = vector.insert %2, %0[0] : f32 into vector<8xf32>
  %4 = constant 1: index
  %5 = load %A[%4] : memref<?xf32>
  %6 = vector.insert %5, %3[1] : f32 into vector<8xf32>
  %7 = constant 2: index
  %8 = load %A[%7] : memref<?xf32>
  %9 = vector.insert %8, %6[2] : f32 into vector<8xf32>
  %10 = constant 3: index
  %11 = load %A[%10] : memref<?xf32>
  %12 = vector.insert %11, %9[3] : f32 into vector<8xf32>
  %13 = constant 4: index
  %14 = load %A[%13] : memref<?xf32>
  %15 = vector.insert %14, %12[4] : f32 into vector<8xf32>
  %16 = constant 5: index
  %17 = load %A[%16] : memref<?xf32>
  %18 = vector.insert %17, %15[5] : f32 into vector<8xf32>
  %19 = constant 6: index
  %20 = load %A[%19] : memref<?xf32>
  %21 = vector.insert %20, %18[6] : f32 into vector<8xf32>
  %22 = constant 7: index
  %23 = load %A[%22] : memref<?xf32>
  %24 = vector.insert %23, %21[7] : f32 into vector<8xf32>
  vector.print %24 : vector<8xf32>
  return
}

func @entry() {
  // Set up memory.
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c8 = constant 8: index
  %A = alloc(%c8) : memref<?xf32>
  scf.for %i = %c0 to %c8 step %c1 {
    %i32 = index_cast %i : index to i32
    %fi = sitofp %i32 : i32 to f32
    store %fi, %A[%i] : memref<?xf32>
  }

  // Set up idx vector.
  %i0 = constant 0: i32
  %i1 = constant 1: i32
  %i2 = constant 2: i32
  %i3 = constant 3: i32
  %i4 = constant 4: i32
  %i5 = constant 5: i32
  %i6 = constant 6: i32
  %i7 = constant 7: i32
  %0 = vector.broadcast %i7 : i32 to vector<8xi32>
  %1 = vector.insert %i0, %0[1] : i32 into vector<8xi32>
  %2 = vector.insert %i1, %1[2] : i32 into vector<8xi32>
  %3 = vector.insert %i6, %2[3] : i32 into vector<8xi32>
  %4 = vector.insert %i2, %3[4] : i32 into vector<8xi32>
  %5 = vector.insert %i4, %4[5] : i32 into vector<8xi32>
  %6 = vector.insert %i5, %5[6] : i32 into vector<8xi32>
  %idx = vector.insert %i3, %6[7] : i32 into vector<8xi32>

  // Set up value vector.
  %f0 = constant 0.0: f32
  %f1 = constant 1.0: f32
  %f2 = constant 2.0: f32
  %f3 = constant 3.0: f32
  %f4 = constant 4.0: f32
  %f5 = constant 5.0: f32
  %f6 = constant 6.0: f32
  %f7 = constant 7.0: f32
  %7 = vector.broadcast %f0 : f32 to vector<8xf32>
  %8 = vector.insert %f1, %7[1] : f32 into vector<8xf32>
  %9 = vector.insert %f2, %8[2] : f32 into vector<8xf32>
  %10 = vector.insert %f3, %9[3] : f32 into vector<8xf32>
  %11 = vector.insert %f4, %10[4] : f32 into vector<8xf32>
  %12 = vector.insert %f5, %11[5] : f32 into vector<8xf32>
  %13 = vector.insert %f6, %12[6] : f32 into vector<8xf32>
  %val = vector.insert %f7, %13[7] : f32 into vector<8xf32>

  // Set up masks.
  %t = constant 1: i1
  %none = vector.constant_mask [0] : vector<8xi1>
  %some = vector.constant_mask [4] : vector<8xi1>
  %more = vector.insert %t, %some[7] : i1 into vector<8xi1>
  %all = vector.constant_mask [8] : vector<8xi1>

  //
  // Scatter tests.
  //

  vector.print %idx : vector<8xi32>
  // CHECK: ( 7, 0, 1, 6, 2, 4, 5, 3 )

  call @printmem(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )

  call @scatter8(%A, %idx, %none, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()

  call @printmem(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 0, 1, 2, 3, 4, 5, 6, 7 )

  call @scatter8(%A, %idx, %some, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()

  call @printmem(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 1, 2, 2, 3, 4, 5, 3, 0 )

  call @scatter8(%A, %idx, %more, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()

  call @printmem(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 1, 2, 2, 7, 4, 5, 3, 0 )

  call @scatter8(%A, %idx, %all, %val)
    : (memref<?xf32>, vector<8xi32>, vector<8xi1>, vector<8xf32>) -> ()

  call @printmem(%A) : (memref<?xf32>) -> ()
  // CHECK: ( 1, 2, 4, 7, 5, 6, 3, 0 )

  return
}
