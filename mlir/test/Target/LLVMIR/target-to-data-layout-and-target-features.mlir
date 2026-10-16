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
// TARGET-FEATURES-SAME: dlti = #dlti.map<"features" = #llvm.target_features<[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-NOT:      +avx
// TARGET-FEATURES-SAME:     +sse
// TARGET-FEATURES-NOT:      +mmx
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = ""
// TARGET-FEATURES-NOT:    features =

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
// TARGET-FEATURES-SAME: dlti = #dlti.map<"features" = #llvm.target_features<[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-NOT:      +avx
// TARGET-FEATURES-SAME:     +mmx
// TARGET-FEATURES-SAME:     +sse
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = ""
// TARGET-FEATURES-SAME:   features = <["+mmx", "+sse"]>

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
// TARGET-FEATURES-SAME: dlti = #dlti.map<"features" = #llvm.target_features<[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-SAME:     +avx
// TARGET-FEATURES-SAME:     +avx2
// TARGET-FEATURES-NOT:      +avx512f
// TARGET-FEATURES-SAME:     +mmx
// TARGET-FEATURES-SAME:     +sse
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = "skylake"
// TARGET-FEATURES-NOT:    features =

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
// TARGET-FEATURES-SAME: dlti = #dlti.map<"features" = #llvm.target_features<[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-NOT:      +avx
// TARGET-FEATURES-NOT:      +avx2
// TARGET-FEATURES-SAME:     +mmx
// TARGET-FEATURES-NOT:      +sse
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = "skylake"
// TARGET-FEATURES-SAME:   features = <["-sse", "-avx"]>

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
// TARGET-FEATURES-SAME: dlti = #dlti.map<
// TARGET-FEATURES-SAME:   "MPI:comm_world_size" = 4 : i64,
// TARGET-FEATURES-SAME:   "features" = #llvm.target_features<[
// TARGET-FEATURES-SAME:     +64bit
// TARGET-FEATURES-SAME:     +avx
// TARGET-FEATURES-SAME:     +avx2
// TARGET-FEATURES-SAME:     +avx512f
// TARGET-FEATURES-NOT:      +mmx
// TARGET-FEATURES-SAME:     +sse
// TARGET-FEATURES-SAME:   "triple" = "preserved"
// TARGET-FEATURES-SAME: #dlti.dl_spec<index = 32 : i64>
// TARGET-FEATURES-SAME: llvm.target = #llvm.target<
// TARGET-FEATURES-SAME:   triple = "x86_64-unknown-linux"
// TARGET-FEATURES-SAME:   chip = "skylake"
// TARGET-FEATURES-SAME:   features = <["-mmx", "+avx512f"]>

module attributes { dlti = #dlti.map<"MPI:comm_world_size" = 4,
                                     "features" = #llvm.target_features<["+old"]>,
                                     "triple" = "preserved">,
                    dlti.dl_spec = #dlti.dl_spec<index = 32>,
                    llvm.target = #llvm.target<triple = "x86_64-unknown-linux",
                                               chip = "skylake",
                                               features = <["-mmx", "+avx512f"]>> } {
}
