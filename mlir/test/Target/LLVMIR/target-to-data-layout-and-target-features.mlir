// REQUIRES: target=x86{{.*}}

// RUN: mlir-opt -llvm-target-to-data-layout -split-input-file %s | FileCheck --check-prefix=DATA-LAYOUT %s
// RUN: mlir-opt -llvm-target-to-target-features -split-input-file %s | FileCheck --check-prefix=TARGET-FEATURES %s

// DATA-LAYOUT: module attributes
// DATA-LAYOUT-SAME: dlti.dl_spec = #dlti.dl_spec
// DATA-LAYOUT-SAME:   "dlti.endianness" = "little"
// DATA-LAYOUT-SAME: llvm.target = #llvm.target<
// DATA-LAYOUT-SAME:   triple = "x86_64-unknown-linux"
// DATA-LAYOUT-SAME:   chip = ""
// DATA-LAYOUT-NOT:    features =

// TARGET-FEATURES: module attributes
// TARGET-FEATURES-NOT:  dlti.dl_spec
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = ""
// TARGET-FEATURES-SAME:   features = <[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-NOT:      +avx
// TARGET-FEATURES-SAME:     +sse
// TARGET-FEATURES-NOT:      +mmx

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = ""> } {
}

// -----

// DATA-LAYOUT: module attributes
// DATA-LAYOUT-SAME: dlti.dl_spec = #dlti.dl_spec
// DATA-LAYOUT-SAME:   "dlti.endianness" = "little"
// DATA-LAYOUT-SAME: llvm.target = #llvm.target<
// DATA-LAYOUT-SAME:   triple = "x86_64-unknown-linux"
// DATA-LAYOUT-SAME:   chip = ""
// DATA-LAYOUT-SAME:   features = <["+mmx", "+sse"]>

// TARGET-FEATURES: module attributes
// TARGET-FEATURES-NOT:  dlti.dl_spec
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = ""
// TARGET-FEATURES-SAME:   features = <[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-NOT:      +avx
// TARGET-FEATURES-SAME:     +mmx
// TARGET-FEATURES-SAME:     +sse

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "",
                                               features = <["+mmx", "+sse"]>> } {
}

// -----

// DATA-LAYOUT: module attributes
// DATA-LAYOUT-SAME: dlti.dl_spec = #dlti.dl_spec
// DATA-LAYOUT-SAME:   "dlti.endianness" = "little"
// DATA-LAYOUT-SAME: llvm.target = #llvm.target<
// DATA-LAYOUT-SAME:   triple = "x86_64-unknown-linux"
// DATA-LAYOUT-SAME:   chip = "skylake"
// DATA-LAYOUT-NOT:    features =

// TARGET-FEATURES: module attributes
// TARGET-FEATURES-NOT:  dlti.dl_spec
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = "skylake"
// TARGET-FEATURES-SAME:   features = <[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-SAME:     +avx
// TARGET-FEATURES-SAME:     +avx2
// TARGET-FEATURES-NOT:      +avx512f
// TARGET-FEATURES-SAME:     +mmx
// TARGET-FEATURES-SAME:     +sse

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake"> } {
}

// -----

// DATA-LAYOUT: module attributes
// DATA-LAYOUT-SAME: dlti.dl_spec = #dlti.dl_spec
// DATA-LAYOUT-SAME:   "dlti.endianness" = "little"
// DATA-LAYOUT-SAME: llvm.target = #llvm.target<
// DATA-LAYOUT-SAME:   triple = "x86_64-unknown-linux"
// DATA-LAYOUT-SAME:   chip = "skylake"
// DATA-LAYOUT-SAME:   features = <["-sse", "-avx"]>

// TARGET-FEATURES: module attributes
// TARGET-FEATURES-NOT:  dlti.dl_spec
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = "skylake"
// TARGET-FEATURES-SAME:   features = <[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-NOT:      +avx
// TARGET-FEATURES-NOT:      +avx2
// TARGET-FEATURES-SAME:     +mmx
// TARGET-FEATURES-NOT:      +sse

module attributes { llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               features = <["-sse", "-avx"]>> } {
}

// -----

// DATA-LAYOUT: module attributes
// DATA-LAYOUT-SAME: dlti.dl_spec = #dlti.dl_spec
// DATA-LAYOUT-SAME:   "dlti.endianness" = "little"
// DATA-LAYOUT-SAME:   index = 32
// DATA-LAYOUT-SAME: llvm.target = #llvm.target<
// DATA-LAYOUT-SAME:   triple = "x86_64-unknown-linux"
// DATA-LAYOUT-SAME:   chip = "skylake"
// DATA-LAYOUT-SAME:   features = <["-mmx", "+avx512f"]>

// TARGET-FEATURES: module attributes
// TARGET-FEATURES-SAME: #dlti.dl_spec<index = 32 : i64>
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = "skylake"
// TARGET-FEATURES-SAME:   features = <[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-SAME:     +avx
// TARGET-FEATURES-SAME:     +avx2
// TARGET-FEATURES-SAME:     +avx512f
// TARGET-FEATURES-NOT:      +mmx
// TARGET-FEATURES-SAME:     +sse

module attributes { dlti.dl_spec = #dlti.dl_spec<index = 32>,
                    llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               features = <["-mmx", "+avx512f"]>> } {
}
