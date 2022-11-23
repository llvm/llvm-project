// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: @binary_ops
// CHECK-SAME: %[[A:.*]]: index, %[[B:.*]]: index
func.func @binary_ops(%a: index, %b: index) {
  // CHECK-NEXT: index.add %[[A]], %[[B]]
  %0 = index.add %a, %b
  // CHECK-NEXT: index.sub %[[A]], %[[B]]
  %1 = index.sub %a, %b
  // CHECK-NEXT: index.mul %[[A]], %[[B]]
  %2 = index.mul %a, %b
  // CHECK-NEXT: index.divs %[[A]], %[[B]]
  %3 = index.divs %a, %b
  // CHECK-NEXT: index.divu %[[A]], %[[B]]
  %4 = index.divu %a, %b
  // CHECK-NEXT: index.ceildivs %[[A]], %[[B]]
  %5 = index.ceildivs %a, %b
  // CHECK-NEXT: index.ceildivu %[[A]], %[[B]]
  %6 = index.ceildivu %a, %b
  // CHECK-NEXT: index.floordivs %[[A]], %[[B]]
  %7 = index.floordivs %a, %b
  // CHECK-NEXT: index.rems %[[A]], %[[B]]
  %8 = index.rems %a, %b
  // CHECK-NEXT: index.remu %[[A]], %[[B]]
  %9 = index.remu %a, %b
  // CHECK-NEXT: index.maxs %[[A]], %[[B]]
  %10 = index.maxs %a, %b
  // CHECK-NEXT: index.maxu %[[A]], %[[B]]
  %11 = index.maxu %a, %b
  // CHECK-NEXT: index.shl %[[A]], %[[B]]
  %12 = index.shl %a, %b
  // CHECK-NEXT: index.shrs %[[A]], %[[B]]
  %13 = index.shrs %a, %b
  // CHECK-NEXT: index.shru %[[A]], %[[B]]
  %14 = index.shru %a, %b
  // CHECK-NEXT: index.and %[[A]], %[[B]]
  %15 = index.and %a, %b
  // CHECK-NEXT: index.or %[[A]], %[[B]]
  %16 = index.or %a, %b
  // CHECK-NEXT: index.xor %[[A]], %[[B]]
  %17 = index.xor %a, %b
  return
}

// CHECK-LABEL: @cmp_op
// CHECK-SAME: %[[A:.*]]: index, %[[B:.*]]: index
func.func @cmp_op(%a: index, %b: index) {
  // CHECK-NEXT: index.cmp eq(%[[A]], %[[B]])
  %0 = index.cmp eq(%a, %b)
  // CHECK-NEXT: index.cmp ne(%[[A]], %[[B]])
  %1 = index.cmp ne(%a, %b)
  // CHECK-NEXT: index.cmp slt(%[[A]], %[[B]])
  %2 = index.cmp slt(%a, %b)
  // CHECK-NEXT: index.cmp sle(%[[A]], %[[B]])
  %3 = index.cmp sle(%a, %b)
  // CHECK-NEXT: index.cmp sgt(%[[A]], %[[B]])
  %4 = index.cmp sgt(%a, %b)
  // CHECK-NEXT: index.cmp sge(%[[A]], %[[B]])
  %5 = index.cmp sge(%a, %b)
  // CHECK-NEXT: index.cmp ult(%[[A]], %[[B]])
  %6 = index.cmp ult(%a, %b)
  // CHECK-NEXT: index.cmp ule(%[[A]], %[[B]])
  %7 = index.cmp ule(%a, %b)
  // CHECK-NEXT: index.cmp ugt(%[[A]], %[[B]])
  %8 = index.cmp ugt(%a, %b)
  // CHECK-NEXT: index.cmp uge(%[[A]], %[[B]])
  %9 = index.cmp uge(%a, %b)
  return
}

// CHECK-LABEL: @sizeof_op
func.func @sizeof_op() {
  // CHECK: index.sizeof
  %0 = index.sizeof
  return
}

// CHECK-LABEL: @constant_op
func.func @constant_op() {
  // CHECK-NEXT: index.constant 0
  %0 = index.constant 0
  // CHECK-NEXT: index.constant 1
  %1 = index.constant 1
  // CHECK-NEXT: index.constant 42
  %2 = index.constant 42
  return
}

// CHECK-LABEL: @bool_constant_op
func.func @bool_constant_op() {
  // CHECK-NEXT: index.bool.constant true
  %0 = index.bool.constant true
  // CHECK-NEXT: index.bool.constant false
  %1 = index.bool.constant false
  return
}

// CHECK-LABEL: @cast_op
// CHECK-SAME: %[[A:.*]]: index, %[[B:.*]]: i32, %[[C:.*]]: i64
func.func @cast_op(%a: index, %b: i32, %c: i64) {
  // CHECK-NEXT: index.casts %[[A]] : index to i64
  %0 = index.casts %a : index to i64
  // CHECK-NEXT: index.casts %[[B]] : i32 to index
  %1 = index.casts %b : i32 to index
  // CHECK-NEXT: index.casts %[[C]] : i64 to index
  %2 = index.casts %c : i64 to index
  // CHECK-NEXT: index.castu %[[A]] : index to i64
  %3 = index.castu %a : index to i64
  // CHECK-NEXT: index.castu %[[B]] : i32 to index
  %4 = index.castu %b : i32 to index
  // CHECK-NEXT: index.castu %[[C]] : i64 to index
  %5 = index.castu %c : i64 to index
  return
}
