// RUN: mlir-opt %s --scf-if-condition-propagation --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: @cond_prop
func.func @cond_prop(%arg0 : i1) -> index {
  %res = scf.if %arg0 -> index {
    %res1 = scf.if %arg0 -> index {
      %v1 = "test.get_some_value1"() : () -> index
      scf.yield %v1 : index
    } else {
      %v2 = "test.get_some_value2"() : () -> index
      scf.yield %v2 : index
    }
    scf.yield %res1 : index
  } else {
    %res2 = scf.if %arg0 -> index {
      %v3 = "test.get_some_value3"() : () -> index
      scf.yield %v3 : index
    } else {
      %v4 = "test.get_some_value4"() : () -> index
      scf.yield %v4 : index
    }
    scf.yield %res2 : index
  }
  return %res : index
}
// CHECK:  %[[if:.+]] = scf.if %arg0 -> (index) {
// CHECK:    %[[c1:.+]] = "test.get_some_value1"() : () -> index
// CHECK:    scf.yield %[[c1]] : index
// CHECK:  } else {
// CHECK:    %[[c4:.+]] = "test.get_some_value4"() : () -> index
// CHECK:    scf.yield %[[c4]] : index
// CHECK:  }
// CHECK:  return %[[if]] : index
// CHECK:}
