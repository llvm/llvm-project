// RUN: mlir-opt --test-data-layout-query %s | FileCheck %s

module attributes { dlti.dl_spec = #dlti.dl_spec<
      #dlti.dl_entry<!test.test_type_with_layout<10>, ["size", 12]>,
      #dlti.dl_entry<!test.test_type_with_layout<20>, ["alignment", 32]>,
      #dlti.dl_entry<!test.test_type_with_layout<30>, ["index", 7]>>} {
  // CHECK-LABEL: @module_level_layout
  func.func @module_level_layout() {
     // CHECK: alignment = 32
     // CHECK: bitsize = 12
     // CHECK: index = 7
     // CHECK: preferred = 1
     // CHECK: size = 2
    "test.data_layout_query"() : () -> !test.test_type_with_layout<10>
    return
  }
}
