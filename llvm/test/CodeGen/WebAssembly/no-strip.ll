; RUN: llc < %s --mtriple=wasm32-unknown-unknown | FileCheck %s

@llvm.used = appending global [
  5 x ptr
] [
  ptr @ga, ptr @gb, ptr @gc, ptr @gd, ptr @ge
], section "llvm.metadata"

; CHECK: .section        .data.ga,"R",@
@ga = global i32 42
; CHECK: .section        .data.gb,"R",@
@gb = internal global i32 41
; CHECK: .section        .data..Lgc,"R",@
@gc = private global i32 40
; CHECK: .section        .rodata.gd,"R",@
@gd = constant i32 39

; All sections with the same explicit name are flagged as retained if a part of them is retained.
; CHECK: .section        dddd,"R",@
@ge = global i32 38, section "dddd"
; CHECK: .section        dddd,"R",@
@gg = global i32 37, section "dddd"
