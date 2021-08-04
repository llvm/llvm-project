; RUN: opt < %s -S -instrprof -runtime-counter-relocation | FileCheck -check-prefixes=RELOC %s

target triple = "x86_64-unknown-linux-gnu"

; RELOC: $__llvm_profile_counter_bias = comdat any
; RELOC: @__llvm_profile_counter_bias = linkonce_odr hidden global i64 0, comdat
