// REQUIRES: target=x86{{.*}}
// RUN: mlir-opt -llvm-target-to-data-layout -split-input-file %s | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec
// CHECK-SAME:   "dlti.endianness" = "little"
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = ""
// CHECK-NOT:    features =

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = ""> } {
}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec
// CHECK-SAME:   "dlti.endianness" = "little"
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = ""
// CHECK-SAME:   features = "+mmx,+sse"

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "",
                                               features = "+mmx,+sse"> } {
}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec
// CHECK-SAME:   "dlti.endianness" = "little"
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = "skylake"
// CHECK-NOT:    features =

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake"> } {
}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec
// CHECK-SAME:   "dlti.endianness" = "little"
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = "skylake"
// CHECK-SAME:   features = "+mmx,+sse">

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               features = "+mmx,+sse"> } {
}

// -----

// CHECK: module attributes
// CHECK-SAME: dlti.dl_spec = #dlti.dl_spec
// CHECK-SAME:   "dlti.endianness" = "little"
// CHECK-SAME:   index = 32
// CHECK-SAME: llvm.target = #llvm.target<
// CHECK-SAME:   triple = "x86_64-unknown-linux"
// CHECK-SAME:   chip = "skylake"
// CHECK-SAME:   features = "+mmx,+sse"

module attributes { dlti.dl_spec = #dlti.dl_spec<index = 32>,
                    llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               features = "+mmx,+sse"> } {
}
