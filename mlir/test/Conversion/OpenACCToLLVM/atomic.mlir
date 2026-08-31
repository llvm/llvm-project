// RUN: mlir-opt %s -acc-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: llvm.func @convert_read
// CHECK: %[[ZERO:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK: %[[RMW:.*]] = llvm.atomicrmw _or %{{.*}}, %[[ZERO]] monotonic : !llvm.ptr, i32
// CHECK: llvm.store %[[RMW]], %{{.*}} : i32, !llvm.ptr

module {
  func.func @convert_read(%v: memref<i32>, %x: memref<i32>) {
    acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @convert_write
// CHECK: llvm.atomicrmw xchg %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32

module {
  func.func @convert_write(%x: memref<i32>, %val: i32) {
    acc.atomic.write %x = %val : memref<i32>, i32
    return
  }
}

// -----

// Normal subtraction: update argument is the LHS, so atomicrmw sub is valid.

// CHECK-LABEL: llvm.func @convert_atomic_sub_lhs
// CHECK: llvm.atomicrmw sub %{{.*}}, %{{.*}} monotonic : !llvm.ptr, i32
// CHECK-NOT: llvm.cmpxchg

module {
  func.func @convert_atomic_sub_lhs(%x: memref<i32>, %val: i32) {
    acc.atomic.update %x : memref<i32> {
    ^bb0(%arg: i32):
      %0 = arith.subi %arg, %val : i32
      acc.yield %0 : i32
    }
    return
  }
}

// -----

// Reversed subtraction: update argument is the RHS. atomicrmw sub would compute
// `*ptr - val`, which is not equivalent, so fall back to cmpxchg.

// CHECK-LABEL: llvm.func @convert_atomic_sub_rhs
// CHECK-NOT: llvm.atomicrmw sub
// CHECK: %[[LOAD:.*]] = llvm.load %{{.*}} : !llvm.ptr -> i32
// CHECK: llvm.br ^bb1(%[[LOAD]] : i32)
// CHECK: ^bb1(%[[LOOP_ARG:.*]]: i32):
// CHECK: %[[SUB:.*]] = llvm.sub %{{.*}}, %[[LOOP_ARG]]
// CHECK: %[[CMPXCHG:.*]] = llvm.cmpxchg %{{.*}}, %[[LOOP_ARG]], %[[SUB]] acq_rel monotonic : !llvm.ptr, i32
// CHECK: %[[NEW_LOADED:.*]] = llvm.extractvalue %[[CMPXCHG]][0]
// CHECK: %[[OK:.*]] = llvm.extractvalue %[[CMPXCHG]][1]
// CHECK: llvm.cond_br %[[OK]], ^bb2, ^bb1(%[[NEW_LOADED]] : i32)

module {
  func.func @convert_atomic_sub_rhs(%x: memref<i32>, %val: i32) {
    acc.atomic.update %x : memref<i32> {
    ^bb0(%arg: i32):
      %0 = arith.subi %val, %arg : i32
      acc.yield %0 : i32
    }
    return
  }
}

// -----

// Test atomic update with max via cmpi/select (no direct atomicrmw mapping).

// CHECK-LABEL: llvm.func @convert_atomic_max
// CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : !llvm.ptr, i32

module {
  func.func @convert_atomic_max(%x: memref<i32>, %val: i32) {
    acc.atomic.update %x : memref<i32> {
    ^bb0(%arg: i32):
      %0 = arith.cmpi sgt, %arg, %val : i32
      %1 = arith.select %0, %arg, %val : i32
      acc.yield %1 : i32
    }
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @convert_capture_rw
// CHECK: %[[XPTR:.*]] = llvm.getelementptr
// CHECK: %[[LOAD:.*]] = llvm.load %[[XPTR]] : !llvm.ptr -> i32
// CHECK: llvm.br ^bb1(%[[LOAD]] : i32)
// CHECK: ^bb1(%[[LOOP_ARG:.*]]: i32):
// CHECK: %[[MUL:.*]] = llvm.mul %[[LOOP_ARG]], %[[LOOP_ARG]] : i32
// CHECK: %{{.*}} = llvm.cmpxchg %[[XPTR]], %[[LOOP_ARG]], %[[MUL]] acq_rel monotonic : !llvm.ptr, i32
// CHECK: llvm.cond_br %{{.*}}, ^bb2, ^bb1(%{{.*}} : i32)

module {
  func.func @convert_capture_rw(%v: memref<i32>, %x: memref<i32>) {
    %0 = memref.load %v[] : memref<i32>
    %1 = arith.muli %0, %0 : i32
    acc.atomic.capture {
      acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
      acc.atomic.write %x = %1 : memref<i32>, i32
    }
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @convert_capture_ru
// CHECK: %[[XPTR:.*]] = llvm.getelementptr
// CHECK: %[[LOAD:.*]] = llvm.load %[[XPTR]] : !llvm.ptr -> i32
// CHECK: llvm.br ^bb1(%[[LOAD]] : i32)
// CHECK: ^bb1(%[[LOOP_ARG:.*]]: i32):
// CHECK: %[[MUL:.*]] = llvm.mul %[[LOOP_ARG]], %[[LOOP_ARG]] : i32
// CHECK: %{{.*}} = llvm.cmpxchg %[[XPTR]], %[[LOOP_ARG]], %[[MUL]] acq_rel monotonic : !llvm.ptr, i32
// CHECK: llvm.cond_br %{{.*}}, ^bb2, ^bb1(%{{.*}} : i32)

module {
  func.func @convert_capture_ru(%v: memref<i32>, %x: memref<i32>) {
    %0 = memref.load %v[] : memref<i32>
    acc.atomic.capture {
      acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
      acc.atomic.update %x : memref<i32> {
      ^bb0(%arg: i32):
        %1 = arith.muli %0, %arg : i32
        acc.yield %1 : i32
      }
    }
    return
  }
}

// -----

// CHECK-LABEL: llvm.func @convert_capture_ur
// CHECK: llvm.cmpxchg %{{.*}}, %{{.*}}, %{{.*}} acq_rel monotonic : !llvm.ptr, i32
// CHECK: llvm.store %{{.*}}, %{{.*}} : i32, !llvm.ptr

module {
  func.func @convert_capture_ur(%v: memref<i32>, %x: memref<i32>, %val: i32) {
    acc.atomic.capture {
      acc.atomic.update %x : memref<i32> {
      ^bb0(%arg: i32):
        %0 = arith.addi %arg, %val : i32
        acc.yield %0 : i32
      }
      acc.atomic.read %v = %x : memref<i32>, memref<i32>, i32
    }
    return
  }
}

// -----

// Test per-component atomicrmw for double complex (complex<f64>) atomic update.

// CHECK-LABEL: llvm.func @double_complex_atomic_add
// CHECK: %[[GEP0:.*]] = llvm.getelementptr %{{.*}}[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64, f64)>
// CHECK: llvm.atomicrmw fadd %[[GEP0]], %{{.*}} monotonic : !llvm.ptr, f64
// CHECK: %[[GEP1:.*]] = llvm.getelementptr %{{.*}}[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64, f64)>
// CHECK: llvm.atomicrmw fadd %[[GEP1]], %{{.*}} monotonic : !llvm.ptr, f64

module {
  func.func @double_complex_atomic_add(%x: memref<complex<f64>>, %val: complex<f64>) {
    %re_val = complex.re %val : complex<f64>
    %im_val = complex.im %val : complex<f64>
    acc.atomic.update %x : memref<complex<f64>> {
    ^bb0(%arg: complex<f64>):
      %re = complex.re %arg : complex<f64>
      %im = complex.im %arg : complex<f64>
      %new_re = arith.addf %re, %re_val : f64
      %new_im = arith.addf %im, %im_val : f64
      %result = complex.create %new_re, %new_im : complex<f64>
      acc.yield %result : complex<f64>
    }
    return
  }
}

// -----

// Two binops on the real lane only must not take the dual-atomicrmw path.

// CHECK-LABEL: llvm.func @double_complex_duplicate_re_lane
// CHECK-NOT: llvm.atomicrmw
// CHECK: llvm.cmpxchg

module {
  func.func @double_complex_duplicate_re_lane(%x: memref<complex<f64>>, %val: complex<f64>) {
    %re_val = complex.re %val : complex<f64>
    %im_val = complex.im %val : complex<f64>
    acc.atomic.update %x : memref<complex<f64>> {
    ^bb0(%arg: complex<f64>):
      %re = complex.re %arg : complex<f64>
      %im = complex.im %arg : complex<f64>
      %a = arith.addf %re, %re_val : f64
      %b = arith.addf %re, %im_val : f64
      %result = complex.create %a, %b : complex<f64>
      acc.yield %result : complex<f64>
    }
    return
  }
}
