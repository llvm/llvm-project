// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Round-tripping the syntax.

// CHECK: #[[MAP:.*]] = affine_map<(d0) -> (d0)>

"test.unknown_op"() {
  // CHECK: #dlti.dl_entry<"test.identifier", 42 : i64>
  test.unknown_attr_1 = #dlti.dl_entry<"test.identifier", 42 : i64>,
  // CHECK: #dlti.dl_entry<"test.identifier", #[[MAP]]>
  test.unknown_attr_2 = #dlti.dl_entry<"test.identifier", affine_map<(d0) -> (d0)>>,
  // CHECK: #dlti.dl_entry<i32, 32 : index>
  test.unknown_attr_3 = #dlti.dl_entry<i32, 32 : index>,
  // CHECK: #dlti.dl_entry<memref<?x?xf32>, ["string", 10]>
  test.unknown_attr_4 = #dlti.dl_entry<memref<?x?xf32>, ["string", 10]>,
  // CHECK: #dlti.dl_spec<>
  test.unknown_attr_5 = #dlti.dl_spec<>,
  // CHECK: #dlti.dl_spec<#dlti.dl_entry<"test.id", 42 : i32>>
  test.unknown_attr_6 = #dlti.dl_spec<#dlti.dl_entry<"test.id", 42 : i32>>,
  // CHECK: #dlti.dl_spec<
  // CHECK:   #dlti.dl_entry<"test.id1", 43 : index>
  // CHECK:   #dlti.dl_entry<"test.id2", 44 : index>
  // CHECK:   #dlti.dl_entry<"test.id3", 45 : index>>
  test.unknown_attr_7 = #dlti.dl_spec<
    #dlti.dl_entry<"test.id1", 43 : index>,
    #dlti.dl_entry<"test.id2", 44 : index>,
    #dlti.dl_entry<"test.id3", 45 : index>>
} : () -> ()

//
// Supported cases where we shouldn't fail. No need to file-check these, not
// triggering an error or an assertion is enough.
//

// Should not fail on missing spec.
"test.op_with_data_layout"() : () -> ()

// Should not fail on empty spec.
"test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<> }: () -> ()

// Should not fail on nested compatible layouts.
"test.op_with_data_layout"() ({
  "test.op_with_data_layout"() { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>> } : () -> ()
  "test.maybe_terminator_op"() : () -> ()
}) { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>> } : () -> ()

// Should not fail on deeper nested compatible layouts.
"test.op_with_data_layout"() ({
  "test.op_with_data_layout"() ({
    "test.op_with_data_layout"()
       { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>> } : () -> ()
    "test.maybe_terminator_op"() : () -> ()
  }) { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>> } : () -> ()
  "test.maybe_terminator_op"() : () -> ()
}) { dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"unknown.unknown", 32>> } : () -> ()

// A valid target system description
// CHECK: module attributes {
// CHECK: dlti.target_system_spec = #dlti.target_system_spec<
// CHECK:  "CPU" : #dlti.target_device_spec<
// CHECK:   #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096 : ui32>>,
// CHECK: "GPU" : #dlti.target_device_spec<
// CHECK:   #dlti.dl_entry<"dlti.max_vector_op_width", 128 : ui32>>
// CHECK: >} {
// CHECK: }
module attributes {
  dlti.target_system_spec = #dlti.target_system_spec<
    "CPU": #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.L1_cache_size_in_bytes", 4096 : ui32>>,
    "GPU": #dlti.target_device_spec<
      #dlti.dl_entry<"dlti.max_vector_op_width", 128 : ui32>>
  >} {}
 
