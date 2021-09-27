// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

// CHECK-LABEL: void opaque_template_args() {
func @opaque_template_args() {
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
