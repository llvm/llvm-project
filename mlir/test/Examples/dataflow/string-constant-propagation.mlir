// RUN: dataflow-opt %s -test-string-constant-propagation -split-input-file | FileCheck %s

//      CHECK: {{.*}} = string.constant "hello "
//      CHECK: {{hello}}
// CHECK-NEXT: {{.*}} = string.constant "world."
//      CHECK: {{world}}
// CHECK-NEXT: {{.*}} = string.concat {{.*}}, {{.*}} :
//      CHECK: {{hello world.}}
func.func @single_concat() {
  %1 = string.constant "hello "
  %2 = string.constant "world."
  %3 = string.concat %1, %2
  return
}

// -----

// CHECK: {{.*}} = string.constant "data"
// CHECK: {{data}}
// CHECK-NEXT: {{.*}} = string.constant "flow "
// CHECK: {{flow}}
// CHECK-NEXT: {{.*}} = string.constant "tutorial" 
// CHECK: {{tutorial}}
// CHECK-NEXT: {{.*}} = string.concat {{.*}}, {{.*}} :
// CHECK: {{dataflow}}
// CHECK-NEXT: {{.*}} = string.concat {{.*}}, {{.*}} :
// CHECK: {{dataflow tutorial}}

func.func @mult_concat() {
  %1 = string.constant "data"
  %2 = string.constant "flow "
  %3 = string.constant "tutorial"
  %4 = string.concat %1, %2
  %5 = string.concat %4, %3
  return
}