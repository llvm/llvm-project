// # RUN: mlir-opt -test-print-invalid %s | FileCheck %s
// # RUN: mlir-opt -test-print-invalid %s --mlir-print-assume-verified  | FileCheck %s --check-prefix=ASSUME-VERIFIED

// The pass creates some ops and prints them to stdout, the input is just an
// empty module.
module {}

// The operation is invalid because the body does not have a terminator, print
// the generic form.
// CHECK:      Invalid operation:
// CHECK-NEXT: "func.func"() ({
// CHECK-NEXT: ^bb0:
// CHECK-NEXT: })
// CHECK-SAME: sym_name = "test"

// The operation is valid because the body has a terminator, print the custom
// form.
// CHECK:      Valid operation:
// CHECK-NEXT: func @test() {
// CHECK-NEXT:   return
// CHECK-NEXT: }

// With --mlir-print-assume-verified the custom form is printed in both cases.
// This works in this particular case, but may crash in general.

// ASSUME-VERIFIED:      Invalid operation:
// ASSUME-VERIFIED-NEXT: func @test() {
// ASSUME-VERIFIED-NEXT: }

// ASSUME-VERIFIED:      Valid operation:
// ASSUME-VERIFIED-NEXT: func @test() {
// ASSUME-VERIFIED-NEXT:   return
// ASSUME-VERIFIED-NEXT: }
