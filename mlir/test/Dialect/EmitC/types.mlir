// RUN: mlir-opt -verify-diagnostics %s | FileCheck %s
// check parser
// RUN: mlir-opt -verify-diagnostics %s | mlir-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @opaque_types() {
func @opaque_types() {
  // CHECK-NEXT: !emitc.opaque<"int">
  emitc.call "f"() {args = [!emitc<"opaque<\"int\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"byte">
  emitc.call "f"() {args = [!emitc<"opaque<\"byte\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"unsigned">
  emitc.call "f"() {args = [!emitc<"opaque<\"unsigned\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"status_t">
  emitc.call "f"() {args = [!emitc<"opaque<\"status_t\">">]} : () -> ()
  // CHECK-NEXT: !emitc.opaque<"std::vector<std::string>">
  emitc.call "f"() {args = [!emitc.opaque<"std::vector<std::string>">]} : () -> ()
  return
}
