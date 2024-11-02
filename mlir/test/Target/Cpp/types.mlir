// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL: void opaque_types() {
func.func @opaque_types() {
  // CHECK-NEXT: f<int>();
  emitc.call "f"() {template_args = [!emitc<"opaque<\"int\">">]} : () -> ()
  // CHECK-NEXT: f<byte>();
  emitc.call "f"() {template_args = [!emitc<"opaque<\"byte\">">]} : () -> ()
  // CHECK-NEXT: f<unsigned>();
  emitc.call "f"() {template_args = [!emitc<"opaque<\"unsigned\">">]} : () -> ()
  // CHECK-NEXT: f<status_t>();
  emitc.call "f"() {template_args = [!emitc<"opaque<\"status_t\">">]} : () -> ()
  // CHECK-NEXT: f<std::vector<std::string>>();
  emitc.call "f"() {template_args = [!emitc.opaque<"std::vector<std::string>">]} : () -> ()

  return
}

// CHECK-LABEL: void ptr_types() {
func.func @ptr_types() {
  // CHECK-NEXT: f<int32_t*>();
  emitc.call "f"() {template_args = [!emitc<"ptr<i32>">]} : () -> ()
  // CHECK-NEXT: f<int64_t*>();
  emitc.call "f"() {template_args = [!emitc.ptr<i64>]} : () -> ()
  // CHECK-NEXT: f<float*>();
  emitc.call "f"() {template_args = [!emitc<"ptr<f32>">]} : () -> ()
  // CHECK-NEXT: f<double*>();
  emitc.call "f"() {template_args = [!emitc.ptr<f64>]} : () -> ()
  // CHECK-NEXT: int32_t* [[V0:[^ ]*]] = f();
  %0 = emitc.call "f"() : () -> (!emitc.ptr<i32>)
  // CHECK-NEXT: int32_t** [[V1:[^ ]*]] = f([[V0:[^ ]*]]);
  %1 = emitc.call "f"(%0) : (!emitc.ptr<i32>) -> (!emitc.ptr<!emitc.ptr<i32>>)
  // CHECK-NEXT: f<int*>();
  emitc.call "f"() {template_args = [!emitc.ptr<!emitc.opaque<"int">>]} : () -> ()

  return
}
