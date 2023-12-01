// RUN: mlir-opt -verify-diagnostics %s | FileCheck %s
// check parser
// RUN: mlir-opt -verify-diagnostics %s | mlir-opt -verify-diagnostics | FileCheck %s

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
