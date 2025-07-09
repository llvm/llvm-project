// RUN: mlir-opt -llvm-data-layout-from-target -split-input-file %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #llvm.data_layout
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-NOT:    chip =
// CHECK-NOT:    #llvm.target_features

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux"> } {
}

// -----

// TODO: module attributes
// TODO-SAME: dlti.dl_spec = #llvm.data_layout
// TODO-SAME: llvm.target = #llvm.target<
// TODO-SAME:   triple = "x86_64-unknown-linux"
// TODO-NOT:   chip = "skylake"
// TODO-SAME:   #llvm.target_features<["+mmx", "+sse"]>
// TODO-NOT:   chip = "skylake"

//module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
//                                               // Generated parser is dumb: expected 'chip'
//                                               #llvm.target_features<["+mmx", "+sse"]>> } {
//}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #llvm.data_layout
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = "skylake"
// CHECK-NOT:    #llvm.target_features

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake"> } {
}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #llvm.data_layout
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = "skylake"
// CHECK-SAME:   #llvm.target_features<["+mmx", "+sse"]>

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               #llvm.target_features<["+mmx", "+sse"]>> } {
}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec<"dlti.endianness" = "little"
// CHECK-SAME:   index = 32
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = "skylake"
// CHECK-SAME:   #llvm.target_features<["+mmx", "+sse"]>

module attributes { dlti.dl_spec = #dlti.dl_spec<index = 32>,
                    llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               #llvm.target_features<["+mmx", "+sse"]>> } {
}
