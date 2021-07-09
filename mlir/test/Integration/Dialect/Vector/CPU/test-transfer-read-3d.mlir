// RUN: mlir-opt %s -convert-vector-to-scf -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf='lower-permutation-maps=true' -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf='full-unroll=true' -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// RUN: mlir-opt %s -convert-vector-to-scf='full-unroll=true lower-permutation-maps=true' -lower-affine -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void  \
// RUN:   -shared-libs=%mlir_integration_test_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @transfer_read_3d(%A : memref<?x?x?x?xf32>,
                       %o: index, %a: index, %b: index, %c: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%o, %a, %b, %c], %fm42
      : memref<?x?x?x?xf32>, vector<2x5x3xf32>
  vector.print %f: vector<2x5x3xf32>
  return
}

func @transfer_read_3d_and_extract(%A : memref<?x?x?x?xf32>,
                                   %o: index, %a: index, %b: index, %c: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%o, %a, %b, %c], %fm42
      {in_bounds = [true, true, true]}
      : memref<?x?x?x?xf32>, vector<2x5x3xf32>
  %sub = vector.extract %f[0] : vector<2x5x3xf32>
  vector.print %sub: vector<5x3xf32>
  return
}

func @transfer_read_3d_broadcast(%A : memref<?x?x?x?xf32>,
                                 %o: index, %a: index, %b: index, %c: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%o, %a, %b, %c], %fm42
      {permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, 0, d3)>}
      : memref<?x?x?x?xf32>, vector<2x5x3xf32>
  vector.print %f: vector<2x5x3xf32>
  return
}

func @transfer_read_3d_mask_broadcast(
    %A : memref<?x?x?x?xf32>, %o: index, %a: index, %b: index, %c: index) {
  %fm42 = constant -42.0: f32
  %mask = constant dense<[0, 1]> : vector<2xi1>
  %f = vector.transfer_read %A[%o, %a, %b, %c], %fm42, %mask
      {permutation_map = affine_map<(d0, d1, d2, d3) -> (d1, 0, 0)>}
      : memref<?x?x?x?xf32>, vector<2x5x3xf32>
  vector.print %f: vector<2x5x3xf32>
  return
}

func @transfer_read_3d_transposed(%A : memref<?x?x?x?xf32>,
                                  %o: index, %a: index, %b: index, %c: index) {
  %fm42 = constant -42.0: f32
  %f = vector.transfer_read %A[%o, %a, %b, %c], %fm42
      {permutation_map = affine_map<(d0, d1, d2, d3) -> (d3, d0, d1)>}
      : memref<?x?x?x?xf32>, vector<3x5x3xf32>
  vector.print %f: vector<3x5x3xf32>
  return
}

func @transfer_write_3d(%A : memref<?x?x?x?xf32>,
                        %o: index, %a: index, %b: index, %c: index) {
  %fn1 = constant -1.0 : f32
  %vf0 = splat %fn1 : vector<2x9x3xf32>
  vector.transfer_write %vf0, %A[%o, %a, %b, %c]
      : vector<2x9x3xf32>, memref<?x?x?x?xf32>
  return
}

func @entry() {
  %c0 = constant 0: index
  %c1 = constant 1: index
  %c2 = constant 2: index
  %c3 = constant 3: index
  %f2 = constant 2.0: f32
  %f10 = constant 10.0: f32
  %first = constant 5: index
  %second = constant 4: index
  %third = constant 2 : index
  %outer = constant 10 : index
  %A = memref.alloc(%outer, %first, %second, %third) : memref<?x?x?x?xf32>
  scf.for %o = %c0 to %outer step %c1 {
    scf.for %i = %c0 to %first step %c1 {
      %i32 = index_cast %i : index to i32
      %fi = sitofp %i32 : i32 to f32
      %fi10 = mulf %fi, %f10 : f32
      scf.for %j = %c0 to %second step %c1 {
        %j32 = index_cast %j : index to i32
        %fj = sitofp %j32 : i32 to f32
        %fadded = addf %fi10, %fj : f32
        scf.for %k = %c0 to %third step %c1 {
          %k32 = index_cast %k : index to i32
          %fk = sitofp %k32 : i32 to f32
          %fk1 = addf %f2, %fk : f32
          %fmul = mulf %fadded, %fk1 : f32
          memref.store %fmul, %A[%o, %i, %j, %k] : memref<?x?x?x?xf32>
        }
      }
    }
  }

  // 1. Read 3D vector from 4D memref.
  call @transfer_read_3d(%A, %c0, %c0, %c0, %c0)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()
  // CHECK: ( ( ( 0, 0, -42 ), ( 2, 3, -42 ), ( 4, 6, -42 ), ( 6, 9, -42 ), ( -42, -42, -42 ) ), ( ( 20, 30, -42 ), ( 22, 33, -42 ), ( 24, 36, -42 ), ( 26, 39, -42 ), ( -42, -42, -42 ) ) )

  // 2. Read 3D vector from 4D memref and extract subvector from result.
  call @transfer_read_3d_and_extract(%A, %c0, %c0, %c0, %c0)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()
  // CHECK: ( ( 0, 0, 2 ), ( 2, 3, 4 ), ( 4, 6, 6 ), ( 6, 9, 20 ), ( 20, 30, 22 ) )

  // 3. Write 3D vector to 4D memref.
  call @transfer_write_3d(%A, %c0, %c0, %c1, %c1)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()

  // 4. Read memref to verify step 2.
  call @transfer_read_3d(%A, %c0, %c0, %c0, %c0)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()
  // CHECK: ( ( ( 0, 0, -42 ), ( 2, -1, -42 ), ( 4, -1, -42 ), ( 6, -1, -42 ), ( -42, -42, -42 ) ), ( ( 20, 30, -42 ), ( 22, -1, -42 ), ( 24, -1, -42 ), ( 26, -1, -42 ), ( -42, -42, -42 ) ) )

  // 5. Read 3D vector from 4D memref and transpose vector.
  call @transfer_read_3d_transposed(%A, %c0, %c0, %c0, %c0)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()
  // CHECK: ( ( ( 0, 20, 40 ), ( 0, 20, 40 ), ( 0, 20, 40 ), ( 0, 20, 40 ), ( 0, 20, 40 ) ), ( ( 0, 30, 60 ), ( 0, 30, 60 ), ( 0, 30, 60 ), ( 0, 30, 60 ), ( 0, 30, 60 ) ), ( ( -42, -42, -42 ), ( -42, -42, -42 ), ( -42, -42, -42 ), ( -42, -42, -42 ), ( -42, -42, -42 ) ) )

  // 6. Read 1D vector from 4D memref and broadcast vector to 3D.
  call @transfer_read_3d_broadcast(%A, %c0, %c0, %c0, %c0)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()
  // CHECK: ( ( ( 0, 0, -42 ), ( 0, 0, -42 ), ( 0, 0, -42 ), ( 0, 0, -42 ), ( 0, 0, -42 ) ), ( ( 20, 30, -42 ), ( 20, 30, -42 ), ( 20, 30, -42 ), ( 20, 30, -42 ), ( 20, 30, -42 ) ) )

  // 7. Read 1D vector from 4D memref with mask and broadcast vector to 3D.
  call @transfer_read_3d_mask_broadcast(%A, %c0, %c0, %c0, %c0)
      : (memref<?x?x?x?xf32>, index, index, index, index) -> ()
  // CHECK: ( ( ( -42, -42, -42 ), ( -42, -42, -42 ), ( -42, -42, -42 ), ( -42, -42, -42 ), ( -42, -42, -42 ) ), ( ( 20, 20, 20 ), ( 20, 20, 20 ), ( 20, 20, 20 ), ( 20, 20, 20 ), ( 20, 20, 20 ) ) )

  return
}
