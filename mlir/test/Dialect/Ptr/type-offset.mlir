// RUN: mlir-opt %s --test-ptr-set-type-offset --split-input-file | FileCheck %s

// CHECK-LABEL: @default_layout
func.func @default_layout() {
  // Include allocation padding, rather than just the bytes storing the value.
  // CHECK: ptr.type_offset i24 {type_offset = 4 : index}
  %0 = ptr.type_offset i24 : index
  // CHECK: ptr.type_offset vector<3xi32> {type_offset = 16 : index}
  %1 = ptr.type_offset vector<3xi32> : index
  // CHECK: ptr.type_offset vector<[4]xi32> {type_offset = 16 : index, type_offset_scalable}
  %2 = ptr.type_offset vector<[4]xi32> : index
  return
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<i24, dense<[64, 128]> : vector<2xi64>>
>} {
  // CHECK-LABEL: @aligned_to_eight
  func.func @aligned_to_eight() {
    // Use the ABI alignment of 8 bytes, not the preferred alignment of 16.
    // CHECK: ptr.type_offset i24 {type_offset = 8 : index}
    %0 = ptr.type_offset i24 : i64
    return
  }

  module @nested {
    // CHECK-LABEL: @inherited_layout
    func.func @inherited_layout() {
      // A scope without a layout specification inherits its enclosing layout.
      // CHECK: ptr.type_offset i24 {type_offset = 8 : index}
      %0 = ptr.type_offset i24 : index
      return
    }
  }
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<
  #dlti.dl_entry<i24, dense<[128, 256]> : vector<2xi64>>
>} {
  // CHECK-LABEL: @aligned_to_sixteen
  func.func @aligned_to_sixteen() {
    // CHECK: ptr.type_offset i24 {type_offset = 16 : index}
    %0 = ptr.type_offset i24 : index
    return
  }
}
