// RUN: mlir-opt -int-range-optimizations %s | FileCheck %s
gpu.module @module{
    gpu.func @kernel_1() kernel {
        %tidx = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
        %tidy = nvvm.read.ptx.sreg.tid.y range <i32, 0, 128> : i32
        %tidz = nvvm.read.ptx.sreg.tid.z range <i32, 0, 4> : i32
        %cidx = nvvm.read.ptx.sreg.cluster.ctaid.x : i32 // unspecified range
        %cond = ub.poison : i1
        %c64 = arith.constant 64 : i32
        
        %1 = arith.cmpi sgt, %tidx, %c64 : i32
        scf.if %1 {
            gpu.printf "threadidx"
        }
        %2 = arith.cmpi sgt, %tidy, %c64 : i32
        scf.if %2 {
            gpu.printf "threadidy"
        }
        %3 = arith.cmpi sgt, %tidz, %c64 : i32
        scf.if %3 {
            gpu.printf "threadidz"
        }
        %4 = arith.select %cond, %cidx, %c64 : i32
        gpu.printf "ctaidx", %4 : i32

        gpu.return
    }
}

// CHECK-LABEL: gpu.func @kernel_1
// CHECK: %[[false:.+]] = arith.constant false
// CHECK: %[[c64_i32:.+]] = arith.constant 64 : i32
// CHECK: %[[S0:.+]] = nvvm.read.ptx.sreg.tid.y range <i32, 0, 128> : i32
// CHECK: scf.if %[[false]] {
// CHECK: gpu.printf "threadidx"
// CHECK: %[[S1:.+]] = arith.cmpi sgt, %[[S0]], %[[c64_i32]] : i32
// CHECK: scf.if %[[S1]] {
// CHECK: gpu.printf "threadidy"
// CHECK: scf.if %[[false]] {
// CHECK: gpu.printf "threadidz"
// CHECK: arith.select

// The exclusive upper bound is not part of either the signed or unsigned range.
// CHECK-LABEL: func.func @exclusive_upper_bound
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: return %[[TRUE]], %[[TRUE]] : i1, i1
func.func @exclusive_upper_bound() -> (i1, i1) {
  %tid = nvvm.read.ptx.sreg.tid.x range <i32, 0, 32> : i32
  %c32 = arith.constant 32 : i32
  %signed = arith.cmpi slt, %tid, %c32 : i32
  %unsigned = arith.cmpi ult, %tid, %c32 : i32
  return %signed, %unsigned : i1, i1
}

// [1, 0x80000000) contains only positive signed i32 values. In particular,
// the exclusive upper bound must not become a negative signed maximum.
// CHECK-LABEL: func.func @positive_grid_dim
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: return %[[FALSE]] : i1
func.func @positive_grid_dim() -> i1 {
  %grid = nvvm.read.ptx.sreg.nctaid.x range <i32, 1, -2147483648> : i32
  %zero = arith.constant 0 : i32
  %negative = arith.cmpi slt, %grid, %zero : i32
  return %negative : i1
}

// CHECK-LABEL: func.func @singleton_range
// CHECK: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: return %[[ONE]] : i32
func.func @singleton_range() -> i32 {
  %dim = nvvm.read.ptx.sreg.ntid.y range <i32, 1, 2> : i32
  return %dim : i32
}

// This interval wraps in unsigned order but has signed bounds [-4, 3].
// CHECK-LABEL: func.func @unsigned_wrapped_range
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[FOUR:.*]] = arith.constant 4 : i32
// CHECK: %[[ID:.*]] = nvvm.read.ptx.sreg.ctaid.x range <i32, -4, 4> : i32
// CHECK: %[[UNSIGNED:.*]] = arith.cmpi ult, %[[ID]], %[[FOUR]] : i32
// CHECK: return %[[TRUE]], %[[TRUE]], %[[UNSIGNED]] : i1, i1, i1
func.func @unsigned_wrapped_range() -> (i1, i1, i1) {
  %id = nvvm.read.ptx.sreg.ctaid.x range <i32, -4, 4> : i32
  %lower = arith.constant -4 : i32
  %upper = arith.constant 4 : i32
  %signed_lower = arith.cmpi sge, %id, %lower : i32
  %signed_upper = arith.cmpi slt, %id, %upper : i32
  %unsigned = arith.cmpi ult, %id, %upper : i32
  return %signed_lower, %signed_upper, %unsigned : i1, i1, i1
}

// This interval crosses the sign bit. Its unsigned bounds are narrow, but
// neither sign can be excluded from its signed interpretation.
// CHECK-LABEL: func.func @signed_wrapped_range
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[ID:.*]] = nvvm.read.ptx.sreg.ctaid.x range <i32, 2147483646, -2147483646> : i32
// CHECK: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[ID]], %[[ZERO]] : i32
// CHECK: return %[[TRUE]], %[[TRUE]], %[[NEGATIVE]] : i1, i1, i1
func.func @signed_wrapped_range() -> (i1, i1, i1) {
  %id = nvvm.read.ptx.sreg.ctaid.x range <i32, 2147483646, -2147483646> : i32
  %lower = arith.constant 2147483646 : i32
  %upper = arith.constant -2147483646 : i32
  %zero = arith.constant 0 : i32
  %unsigned_lower = arith.cmpi uge, %id, %lower : i32
  %unsigned_upper = arith.cmpi ult, %id, %upper : i32
  %negative = arith.cmpi slt, %id, %zero : i32
  return %unsigned_lower, %unsigned_upper, %negative : i1, i1, i1
}

// A full range must not be mistaken for the constant -1.
// CHECK-LABEL: func.func @full_range
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[ID:.*]] = nvvm.read.ptx.sreg.ctaid.x range <i32, -1, -1> : i32
// CHECK: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[ID]], %[[ZERO]] : i32
// CHECK: return %[[NEGATIVE]] : i1
func.func @full_range() -> i1 {
  %id = nvvm.read.ptx.sreg.ctaid.x range <i32, -1, -1> : i32
  %zero = arith.constant 0 : i32
  %negative = arith.cmpi slt, %id, %zero : i32
  return %negative : i1
}

// Empty sets have no ConstantIntRanges representation. Inference falls back
// to the maximum range rather than constructing inconsistent signed bounds.
// CHECK-LABEL: func.func @empty_range
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[ID:.*]] = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 0> : i32
// CHECK: %[[NEGATIVE:.*]] = arith.cmpi slt, %[[ID]], %[[ZERO]] : i32
// CHECK: return %[[NEGATIVE]] : i1
func.func @empty_range() -> i1 {
  %id = nvvm.read.ptx.sreg.ctaid.x range <i32, 0, 0> : i32
  %zero = arith.constant 0 : i32
  %negative = arith.cmpi slt, %id, %zero : i32
  return %negative : i1
}
