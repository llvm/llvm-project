// RUN: mlir-opt %s -lower-affine -convert-scf-to-cf -convert-vector-to-llvm="enable-arm-sve" -convert-memref-to-llvm -convert-func-to-llvm -convert-arith-to-llvm -canonicalize | \
// RUN: mlir-translate -mlir-to-llvmir | \
// RUN: %lli --entry-function=entry --march=aarch64 --mattr="+sve" --dlopen=%mlir_native_utils_lib_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

// Note: To run this test, your CPU must support SVE

// VLA memcopy
func @kernel_copy(%src : memref<?xi64>, %dst : memref<?xi64>, %size : index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %vs = vector.vscale
  %step = arith.muli %c2, %vs : index
  scf.for %i0 = %c0 to %size step %step {
    %0 = vector.load %src[%i0] : memref<?xi64>, vector<[2]xi64>
    vector.store %0, %dst[%i0] : memref<?xi64>, vector<[2]xi64>
  }

  return
}

// VLA multiply and add
func @kernel_muladd(%a : memref<?xi64>,
                    %b : memref<?xi64>,
                    %c : memref<?xi64>,
                    %size : index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %vs = vector.vscale
  %step = arith.muli %c2, %vs : index
  scf.for %i0 = %c0 to %size step %step {
    %0 = vector.load %a[%i0] : memref<?xi64>, vector<[2]xi64>
    %1 = vector.load %b[%i0] : memref<?xi64>, vector<[2]xi64>
    %2 = vector.load %c[%i0] : memref<?xi64>, vector<[2]xi64>
    %3 = arith.muli %0, %1 : vector<[2]xi64>
    %4 = arith.addi %3, %2 : vector<[2]xi64>
    vector.store %4, %c[%i0] : memref<?xi64>, vector<[2]xi64>
  }
  return
}

// SVE-based absolute difference
func @kernel_absdiff(%a : memref<?xi64>,
                     %b : memref<?xi64>,
                     %c : memref<?xi64>,
                     %size : index) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %vs = vector.vscale
  %step = arith.muli %c2, %vs : index
  scf.for %i0 = %c0 to %size step %step {
    %0 = vector.load %a[%i0] : memref<?xi64>, vector<[2]xi64>
    %1 = vector.load %b[%i0] : memref<?xi64>, vector<[2]xi64>
    %agb = arith.cmpi sge, %0, %1 : vector<[2]xi64>
    %bga = arith.cmpi slt, %0, %1 : vector<[2]xi64>
    %10 = arm_sve.masked.subi %agb, %0, %1 : vector<[2]xi1>,
                                             vector<[2]xi64>
    %01 = arm_sve.masked.subi %bga, %1, %0 : vector<[2]xi1>,
                                             vector<[2]xi64>
    vector.maskedstore %c[%i0], %agb, %10 : memref<?xi64>,
                                            vector<[2]xi1>,
                                            vector<[2]xi64>
    vector.maskedstore %c[%i0], %bga, %01 : memref<?xi64>,
                                            vector<[2]xi1>,
                                            vector<[2]xi64>
  }
  return
}

// VLA unknown bounds vector addition
func @kernel_addition(%a : memref<?xf32>,
                      %b : memref<?xf32>,
                      %c : memref<?xf32>,
                      %N : index) {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 2 : index
  %v0f = arith.constant dense<0.0> : vector<[4]xf32>
  %vs = vector.vscale
  %s = arith.muli %c4, %vs : index
  scf.for %i0 = %c0 to %N step %s {
    %sub = affine.min affine_map<(d0, d1)[s0] -> (s0, d0 - d1)>(%N, %i0)[%s]
    %mask = vector.create_mask %sub : vector<[4]xi1>
    %la = vector.maskedload %a[%i0], %mask, %v0f : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
    %lb = vector.maskedload %b[%i0], %mask, %v0f : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32> into vector<[4]xf32>
    %lc = arith.addf %la, %lb : vector<[4]xf32>
    vector.maskedstore %c[%i0], %mask, %lc : memref<?xf32>, vector<[4]xi1>, vector<[4]xf32>
  }
  return
}

func @entry() -> i32 {
  %i0 = arith.constant 0: i64
  %i1 = arith.constant 1: i64
  %r0 = arith.constant 0: i32
  %f0 = arith.constant 0.0: f32
  %c0 = arith.constant 0: index
  %c1 = arith.constant 1: index
  %c2 = arith.constant 2: index
  %c4 = arith.constant 4: index
  %c8 = arith.constant 8: index
  %c32 = arith.constant 32: index
  %c33 = arith.constant 33: index
  %c-1 = arith.constant -1.0 : f32

  // Set up memory.
  %a = memref.alloc()      : memref<32xi64>
  %a_copy = memref.alloc() : memref<32xi64>
  %b = memref.alloc()      : memref<32xi64>
  %c = memref.alloc()      : memref<32xi64>
  %d = memref.alloc()      : memref<32xi64>
  %e = memref.alloc()      : memref<33xf32>
  %f = memref.alloc()      : memref<33xf32>
  %g = memref.alloc()      : memref<36xf32>

  %a_data = arith.constant dense<[1 , 2,  3 , 4 , 5,  6,  7,  8,
                                9, 10, 11, 12, 13, 14, 15, 16,
                                17, 18, 19, 20, 21, 22, 23, 24,
                                25, 26, 27, 28, 29, 30, 31, 32]> : vector<32xi64>
  vector.transfer_write %a_data, %a[%c0] : vector<32xi64>, memref<32xi64>
  %b_data = arith.constant dense<[33, 34, 35, 36, 37, 38, 39, 40,
                                41, 42, 43, 44, 45, 46, 47, 48,
                                49, 50, 51, 52, 53, 54, 55, 56,
                                57, 58, 59, 60, 61, 62, 63, 64]> : vector<32xi64>
  vector.transfer_write %b_data, %b[%c0] : vector<32xi64>, memref<32xi64>
  %d_data = arith.constant dense<[-9, 76, -7, 78, -5, 80, -3, 82,
                                -1, 84, 1, 86, 3, 88, 5, 90,
                                7, 92, 9, 94, 11, 96, 13, 98,
                                15, 100, 17, 102, 19, 104, 21, 106]> : vector<32xi64>
  vector.transfer_write %d_data, %d[%c0] : vector<32xi64>, memref<32xi64>
  %zero_data = vector.broadcast %i0 : i64 to vector<32xi64>
  vector.transfer_write %zero_data, %a_copy[%c0] : vector<32xi64>, memref<32xi64>
  %one_data = vector.broadcast %i1 : i64 to vector<32xi64>
  vector.transfer_write %one_data, %c[%c0] : vector<32xi64>, memref<32xi64>

  %e_data = arith.constant dense<[1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5,
                                9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5,
                                17.5, 18.5, 19.5, 20.5, 21.5, 22.5, 23.5, 24.5,
                                25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 31.5, 32.5,
                                33.5]> : vector<33xf32>
  vector.transfer_write %e_data, %e[%c0] : vector<33xf32>, memref<33xf32>
  %f_data = arith.constant dense<[40.5, 39.5, 38.5, 37.5, 36.5, 35.5, 34.5, 33.5,
                                32.5, 31.5, 30.5, 29.5, 28.5, 27.5, 26.5, 25.5,
                                24.5, 23.5, 22.5, 21.5, 20.5, 19.5, 18.5, 17.5,
                                16.5, 15.5, 14.5, 13.5, 12.5, 11.5, 10.5, 9.5,
                                8.5]> : vector<33xf32>
  vector.transfer_write %f_data, %f[%c0] : vector<33xf32>, memref<33xf32>
  %minus1_data = vector.broadcast %c-1 : f32 to vector<36xf32>
  vector.transfer_write %minus1_data, %g[%c0] : vector<36xf32>, memref<36xf32>

  // Call kernel.
  %0 = memref.cast %a : memref<32xi64> to memref<?xi64>
  %1 = memref.cast %a_copy : memref<32xi64> to memref<?xi64>
  call @kernel_copy(%0, %1, %c32) : (memref<?xi64>, memref<?xi64>, index) -> ()

  // Print and verify.
  //
  // CHECK:      ( 1, 2, 3, 4 )
  // CHECK-NEXT: ( 5, 6, 7, 8 )
  scf.for %i = %c0 to %c32 step %c4 {
    %cv = vector.transfer_read %a_copy[%i], %i0: memref<32xi64>, vector<4xi64>
    vector.print %cv : vector<4xi64>
  }

  %2 = memref.cast %a : memref<32xi64> to memref<?xi64>
  %3 = memref.cast %b : memref<32xi64> to memref<?xi64>
  %4 = memref.cast %c : memref<32xi64> to memref<?xi64>
  call @kernel_muladd(%2, %3, %4, %c32) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, index) -> ()

  // CHECK:      ( 34, 69, 106, 145 )
  // CHECK-NEXT: ( 186, 229, 274, 321 )
  scf.for %i = %c0 to %c32 step %c4 {
    %macv = vector.transfer_read %c[%i], %i0: memref<32xi64>, vector<4xi64>
    vector.print %macv : vector<4xi64>
  }

  %5 = memref.cast %b : memref<32xi64> to memref<?xi64>
  %6 = memref.cast %d : memref<32xi64> to memref<?xi64>
  %7 = memref.cast %c : memref<32xi64> to memref<?xi64>
  call @kernel_absdiff(%5, %6, %7, %c32) : (memref<?xi64>, memref<?xi64>, memref<?xi64>, index) -> ()

  // CHECK:      ( 42, 42, 42, 42 )
  // CHECK-NEXT: ( 42, 42, 42, 42 )
  scf.for %i = %c0 to %c32 step %c4 {
    %abdv = vector.transfer_read %c[%i], %i0: memref<32xi64>, vector<4xi64>
    vector.print %abdv : vector<4xi64>
  }

  %ee = memref.cast %e : memref<33xf32> to memref<?xf32>
  %ff = memref.cast %f : memref<33xf32> to memref<?xf32>
  %gg = memref.cast %g : memref<36xf32> to memref<?xf32>
  call @kernel_addition(%ee, %ff, %gg, %c33) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index) -> ()

  // CHECK:      ( 42, 42, 42, 42, 42, 42, 42, 42 )
  // CHECK-NEXT: ( 42, 42, 42, 42, 42, 42, 42, 42 )
  // CHECK-NEXT: ( 42, 42, 42, 42, 42, 42, 42, 42 )
  // CHECK-NEXT: ( 42, 42, 42, 42, 42, 42, 42, 42 )
  // CHECK-NEXT: ( 42, -1, -1, -1 )
  scf.for %i = %c0 to %c32 step %c8 {
    %addv = vector.transfer_read %g[%i], %f0: memref<36xf32>, vector<8xf32>
    vector.print %addv : vector<8xf32>
  }
  %remv = vector.transfer_read %g[%c32], %f0: memref<36xf32>, vector<4xf32>
  vector.print %remv : vector<4xf32>

  // Release resources.
  memref.dealloc %a      : memref<32xi64>
  memref.dealloc %a_copy : memref<32xi64>
  memref.dealloc %b      : memref<32xi64>
  memref.dealloc %c      : memref<32xi64>
  memref.dealloc %d      : memref<32xi64>
  memref.dealloc %e      : memref<33xf32>
  memref.dealloc %f      : memref<33xf32>
  memref.dealloc %g      : memref<36xf32>

  return %r0 : i32
}
