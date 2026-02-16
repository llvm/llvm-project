// RUN: mlir-opt --convert-to-llvm="filter-dialects=func dynamic=true" --split-input-file %s


// CHECK-LABEL: llvm.func @test_default_index
// CHECK-SAME: (%{{.*}}: i64) -> i64
func.func private @test_default_index(%arg0: index) -> index

// -----

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<
// CHECK-SAME: #dlti.dl_entry<index, 32>
// CHECK-SAME: >}
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32>>} {
  // CHECK-LABEL: llvm.func @test_32bit_index
  // CHECK-SAME: (%{{.*}}: i32) -> i32
  func.func private @test_32bit_index(%arg0: index) -> index
}

// -----

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<
// CHECK-SAME: #dlti.dl_entry<index, 64>
// CHECK-SAME: >}
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64>>} {
  // CHECK-LABEL: llvm.func @test_64bit_index
  // CHECK-SAME: (%{{.*}}: i64) -> i64
  func.func private @test_64bit_index(%arg0: index) -> index
}

// -----

// CHECK-LABEL: module attributes {dlti.dl_spec = #dlti.dl_spec<
// CHECK-SAME: #dlti.dl_entry<index, 16>
// CHECK-SAME: >}
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 16>>} {
  // CHECK-LABEL: llvm.func @test_16bit_index
  // CHECK-SAME: (%{{.*}}: i16) -> i16
  func.func private @test_16bit_index(%arg0: index) -> index
}
