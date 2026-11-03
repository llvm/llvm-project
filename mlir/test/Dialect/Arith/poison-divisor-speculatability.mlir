// RUN: mlir-opt %s -loop-invariant-code-motion | FileCheck %s

// A poison divisor must not allow division or remainder to leave a loop.

// CHECK-LABEL: @divui(
// CHECK: scf.for
// CHECK: arith.divui
// CHECK: memref.store
func.func @divui(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.divui %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @divui_zero_trip(
// CHECK: scf.for
// CHECK: arith.divui
// CHECK: memref.store
func.func @divui_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.divui %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @divsi(
// CHECK: scf.for
// CHECK: arith.divsi
// CHECK: memref.store
func.func @divsi(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.divsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @divsi_zero_trip(
// CHECK: scf.for
// CHECK: arith.divsi
// CHECK: memref.store
func.func @divsi_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.divsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @ceildivui(
// CHECK: scf.for
// CHECK: arith.ceildivui
// CHECK: memref.store
func.func @ceildivui(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.ceildivui %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @ceildivui_zero_trip(
// CHECK: scf.for
// CHECK: arith.ceildivui
// CHECK: memref.store
func.func @ceildivui_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.ceildivui %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @ceildivsi(
// CHECK: scf.for
// CHECK: arith.ceildivsi
// CHECK: memref.store
func.func @ceildivsi(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.ceildivsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @ceildivsi_zero_trip(
// CHECK: scf.for
// CHECK: arith.ceildivsi
// CHECK: memref.store
func.func @ceildivsi_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.ceildivsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @floordivsi(
// CHECK: scf.for
// CHECK: arith.floordivsi
// CHECK: memref.store
func.func @floordivsi(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.floordivsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @floordivsi_zero_trip(
// CHECK: scf.for
// CHECK: arith.floordivsi
// CHECK: memref.store
func.func @floordivsi_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.floordivsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @remui(
// CHECK: scf.for
// CHECK: arith.remui
// CHECK: memref.store
func.func @remui(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.remui %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @remui_zero_trip(
// CHECK: scf.for
// CHECK: arith.remui
// CHECK: memref.store
func.func @remui_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.remui %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @remsi(
// CHECK: scf.for
// CHECK: arith.remsi
// CHECK: memref.store
func.func @remsi(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %n step %one {
    %r = arith.remsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}

// CHECK-LABEL: @remsi_zero_trip(
// CHECK: scf.for
// CHECK: arith.remsi
// CHECK: memref.store
func.func @remsi_zero_trip(%x: i32, %n: index, %m: memref<?xi32>) {
  %p = ub.poison : i32
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  scf.for %i = %zero to %zero step %one {
    %r = arith.remsi %x, %p : i32
    memref.store %r, %m[%i] : memref<?xi32>
  }
  return
}
