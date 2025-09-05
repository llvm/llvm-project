// RUN: mlir-opt %s | FileCheck %s
// check parser
// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func @opaque_attrs() {
func.func @opaque_attrs() {
  // CHECK-NEXT: #emitc.opaque<"attr">
  emitc.call_opaque "f"() {args = [#emitc.opaque<"attr">]} : () -> ()
  // CHECK-NEXT: #emitc.opaque<"\22quoted_attr\22">
  emitc.call_opaque "f"() {args = [#emitc.opaque<"\"quoted_attr\"">]} : () -> ()
  return
}
