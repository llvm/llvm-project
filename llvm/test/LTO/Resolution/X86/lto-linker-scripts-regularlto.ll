; RUN: opt -module-summary -passes=assign-sections-to-globals %s -o %t
; RUN: rm -f %t2.0 %t2.0.5.precodegen.bc
; RUN: llvm-lto2 run %t -O0 -r %t,foo,pxofoo -r %t,g,pxobar -o %t2 -select-save-temps=precodegen -include-local-symbols
; RUN: llvm-objdump --syms %t2.0 | FileCheck %s
; RUN: opt -S %t2.0.5.precodegen.bc | FileCheck -check-prefix=IR %s

; CHECK: 0000000000000000 l O .bss^^{{.*}}lto-linker-scripts-regularlto.ll.tmp 0000000000000004 g
; CHECK: 0000000000000000 g F .text^^{{.*}}lto-linker-scripts-regularlto.ll.tmp 0000000000000008 foo

; IR: @g = internal global i32 0, section ".bss^^{{.*}}", !guid !0 #0
; IR: define ptr @foo() #1 section ".text^^{{.*}}" !guid !2 {
; IR: attributes #0 = { "linker_output_section"="bar" }
; IR: attributes #1 = { "linker_output_section"="foo" }

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = internal global i32 0

define ptr @foo(){
  ret ptr @g
}

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"ThinLTO", i32 0}
