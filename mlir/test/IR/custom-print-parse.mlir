// RUN: mlir-opt %s | FileCheck %s

// CHECK-LABEL: module @dimension_list
module @dimension_list {
  // CHECK: test.custom_dimension_list_attr dimension_list = []
  test.custom_dimension_list_attr dimension_list = []
  // CHECK: test.custom_dimension_list_attr dimension_list = 3
  test.custom_dimension_list_attr dimension_list = 3
  // CHECK: test.custom_dimension_list_attr dimension_list = 0
  test.custom_dimension_list_attr dimension_list = 0
  // CHECK: test.custom_dimension_list_attr dimension_list = 1x2
  test.custom_dimension_list_attr dimension_list = 1x2
  // CHECK: test.custom_dimension_list_attr dimension_list = ?
  test.custom_dimension_list_attr dimension_list = ?
  // CHECK: test.custom_dimension_list_attr dimension_list = ?x?
  test.custom_dimension_list_attr dimension_list = ?x?
}
