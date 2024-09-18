// RUN: mlir-opt -verify-diagnostics -allow-unregistered-dialect %s | FileCheck %s
// check parser
// RUN: mlir-opt -verify-diagnostics -allow-unregistered-dialect %s | mlir-opt -verify-diagnostics --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func @array_types(
func.func @array_types(
  // CHECK-SAME: !emitc.array<1xf32>,
  %arg0: !emitc.array<1xf32>,
  // CHECK-SAME: !emitc.array<10x20x30xi32>,
  %arg1: !emitc.array<10x20x30xi32>,
  // CHECK-SAME: !emitc.array<30x!emitc.ptr<i32>>,
  %arg2: !emitc.array<30x!emitc.ptr<i32>>,
  // CHECK-SAME: !emitc.array<30x!emitc.opaque<"int">>
  %arg3: !emitc.array<30x!emitc.opaque<"int">>,
  // CHECK-SAME: !emitc.array<30x!emitc.size_t>
  %arg4: !emitc.array<30x!emitc.size_t>,
  // CHECK-SAME: !emitc.array<30x!emitc.ssize_t>
  %arg5: !emitc.array<30x!emitc.ssize_t>,
  // CHECK-SAME: !emitc.array<30x!emitc.ptrdiff_t>
  %arg6: !emitc.array<30x!emitc.ptrdiff_t>
) {
  return
}

// CHECK-LABEL: func @lvalue_types(
func.func @lvalue_types() {
  // CHECK-NEXT: !emitc.lvalue<i32>
  %0 = "typed.result"() : () -> (!emitc.lvalue<i32>)
  // CHECK-NEXT: !emitc.lvalue<f64>
  %2 = "typed.result"() : () -> (!emitc.lvalue<f64>)
  // CHECK-NEXT: !emitc.lvalue<index>
  %3 = "typed.result"() : () -> (!emitc.lvalue<index>)
  // CHECK-NEXT: !emitc.lvalue<!emitc.ptr<i32>>
  %4 = "typed.result"() : () -> (!emitc.lvalue<!emitc.ptr<i32>>)
  // CHECK-NEXT: !emitc.lvalue<!emitc.opaque<"int">>
  %5 = "typed.result"() : () -> (!emitc.lvalue<!emitc.opaque<"int">>)
  return
}

// CHECK-LABEL: func @opaque_types() {
func.func @opaque_types() {
  // CHECK-NEXT: !emitc.opaque<"int">
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"int">>]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"byte">
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"byte">>]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"unsigned">
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"unsigned">>]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"status_t">
  emitc.call_opaque "f"() {template_args = [!emitc<opaque<"status_t">>]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"std::vector<std::string>">
  emitc.call_opaque "f"() {template_args = [!emitc.opaque<"std::vector<std::string>">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"SmallVector<int*, 4>">
  emitc.call_opaque "f"() {template_args = [!emitc.opaque<"SmallVector<int*, 4>">]} : () -> ()

  return
}

// CHECK-LABEL: func @pointer_types() {
func.func @pointer_types() {
  // CHECK-NEXT: !emitc.ptr<i32>
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<i32>]} : () -> ()
  // CHECK-NEXT: !emitc.ptr<i64>
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<i64>]} : () -> ()
  // CHECK-NEXT: !emitc.ptr<f32>
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<f32>]} : () -> ()
  // CHECK-NEXT: !emitc.ptr<f64>
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<f64>]} : () -> ()
  // CHECK-NEXT: !emitc.ptr<i32>
  %0 = emitc.call_opaque "f"() : () -> (!emitc.ptr<i32>)
  // CHECK-NEXT: (!emitc.ptr<i32>) -> !emitc.ptr<!emitc.ptr<i32>>
  %1 = emitc.call_opaque "f"(%0) : (!emitc.ptr<i32>) -> (!emitc.ptr<!emitc.ptr<i32>>)
  // CHECK-NEXT: !emitc.ptr<!emitc.opaque<"int">>
  emitc.call_opaque "f"() {template_args = [!emitc.ptr<!emitc.opaque<"int">>]} : () -> ()

  return
}

// CHECK-LABEL: func @size_types() 
func.func @size_types() {
  // CHECK-NEXT: !emitc.ssize_t
  emitc.call_opaque "f"() {template_args = [!emitc.ssize_t]} : () -> ()
  // CHECK-NEXT: !emitc.size_t
  emitc.call_opaque "f"() {template_args = [!emitc.size_t]} : () -> ()
  // CHECK-NEXT: !emitc.ptrdiff_t
  emitc.call_opaque "f"() {template_args = [!emitc.ptrdiff_t]} : () -> ()

  return
}
