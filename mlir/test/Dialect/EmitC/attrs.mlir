// RUN: mlir-opt -verify-diagnostics %s | FileCheck %s
// check parser
// RUN: mlir-opt -verify-diagnostics %s | mlir-opt -verify-diagnostics | FileCheck %s

// CHECK-LABEL: func @opaque_attrs() {
func @opaque_attrs() {
  // CHECK-NEXT: #emitc.opaque<"attr">
  emitc.call "f"() {args = [#emitc.opaque<"attr">]} : () -> ()
  // CHECK-NEXT: #emitc.opaque<"\22quoted_attr\22">
  emitc.call "f"() {args = [#emitc.opaque<"\"quoted_attr\"">]} : () -> ()
  return
}
