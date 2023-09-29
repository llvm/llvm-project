// RUN: mlir-opt -allow-unregistered-dialect -verify-diagnostics -split-input-file %s | FileCheck %s

#array = [#integer_attr, !integer_type]
!integer_type = i32
#integer_attr = 8 : !integer_type

// CHECK-LABEL: func @foo()
func.func @foo() {
  // CHECK-NEXT: value = [8 : i32, i32]
  "foo.attr"() { value = #array} : () -> ()
}

// -----

// Check that only groups may reference later defined aliases.

// expected-error@below {{undefined symbol alias id 'integer_attr'}}
#array = [!integer_type, #integer_attr]
!integer_type = i32

func.func @foo() {
  %0 = "foo.attr"() { value = #array}
}

#integer_attr = 8 : !integer_type
