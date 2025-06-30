// RUN: mlir-opt --wrap-emitc-func-in-class %s | FileCheck %s

emitc.func @foo(%arg0 : !emitc.array<1xf32>) {
  emitc.call_opaque "bar" (%arg0) : (!emitc.array<1xf32>) -> ()
  emitc.return
}

// CHECK: module {
// CHECK-NEXT:   emitc.class @fooClass {
// CHECK-NEXT:     emitc.field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:     emitc.func @execute() {
// CHECK-NEXT:       %0 = get_field @fieldName0 : !emitc.array<1xf32>
// CHECK-NEXT:       call_opaque "bar"(%0) : (!emitc.array<1xf32>) -> ()
// CHECK-NEXT:       return
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: }
