// Verifier errors highlight the directive.

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: not mlir-opt --verify-diagnostics input.mlir 2>&1 \
// RUN:   | FileCheck %s --strict-whitespace

// FIXME: The highlight ends in the middle of the directive.
// CHECK:      input.mlir:1:4: error: expected error "never produced" was not produced
// CHECK-NEXT: {{^}}// expected-error {{[{][{]}}never produced{{[}][}]}}{{$}}
// CHECK-NEXT: {{^}}   ^~~~~~~~~~~~~~{{$}}

//--- input.mlir
// expected-error {{never produced}}
