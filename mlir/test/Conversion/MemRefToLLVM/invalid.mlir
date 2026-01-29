// RUN: mlir-opt %s -finalize-memref-to-llvm -split-input-file -verify-diagnostics | FileCheck %s

// expected-error@+1{{redefinition of reserved function 'malloc' of different type '!llvm.func<void (i64)>' is prohibited}}
llvm.func @malloc(i64)
func.func @redef_reserved() {
    %alloc = memref.alloc() : memref<1024x64xf32, 1>
    llvm.return
}

// -----

// expected-error@unknown{{conversion of memref memory space "foo" to integer address space failed. Consider adding memory space conversions.}}
// CHECK-LABEL: @bad_address_space
func.func @bad_address_space(%a: memref<2xindex, "foo">) {
    %c0 = arith.constant 0 : index
    // CHECK: memref.store
    memref.store %c0, %a[%c0] : memref<2xindex, "foo">
    return
}

// -----

// CHECK-LABEL: @invalid_int_conversion
func.func @invalid_int_conversion() {
     // expected-error@unknown{{conversion of memref memory space 1 : ui64 to integer address space failed. Consider adding memory space conversions.}}
     %alloc = memref.alloc() {alignment = 64 : i64} : memref<10xf32, 1 : ui64> 
    return
}

// -----

// expected-error@unknown{{conversion of memref memory space #gpu.address_space<workgroup> to integer address space failed. Consider adding memory space conversions}}
// CHECK-LABEL: @issue_70160
func.func @issue_70160() {
  %alloc = memref.alloc() : memref<1x32x33xi32, #gpu.address_space<workgroup>>
  %alloc1 = memref.alloc() : memref<i32>
  %c0 = arith.constant 0 : index
  // CHECK: memref.load
  %0 = memref.load %alloc[%c0, %c0, %c0] : memref<1x32x33xi32, #gpu.address_space<workgroup>>
  memref.store %0, %alloc1[] : memref<i32>
  func.return
}


// -----

func.func @test_atomic_exch(%arg0: memref<?xi32>, %idx: index, %value: i32) {
  // expected-error @+1 {{result not defined in region}}
  %1 = memref.generic_atomic_rmw %arg0[%idx] : memref<?xi32> {
  ^bb0(%arg3: i32):
    memref.atomic_yield %value : i32
  }
  func.return
}

// -----

func.func @generic_atomic_rmw_rank_mismatch(%arg0: memref<i32>, %idx: index) {
  // expected-error@+1 {{index count (1) does not match memref rank (0)}}
  %r = memref.generic_atomic_rmw %arg0[%idx] : memref<i32> {
  ^bb0(%v: i32):
    memref.atomic_yield %v : i32
  }
  func.return
}
