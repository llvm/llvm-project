; ThinLTO
; RUN: opt -module-summary -passes=assign-sections-to-globals %s -o %t
; RUN: rm -f %t2.1 %t2.1.5.precodegen.bc %t2.index.bc
; RUN: llvm-lto2 run %t -O0 -r %t,foo,pxofoo -r %t,g,pxobar -o %t2 -select-save-temps=precodegen,combinedindex -include-local-symbols
; RUN: llvm-objdump --syms %t2.1 | FileCheck %s
; RUN: opt -S %t2.1.5.precodegen.bc | FileCheck -check-prefix=IR %s
; RUN: llvm-dis -o - %t2.index.bc | FileCheck -check-prefix=INDEX %s

; CHECK: 0000000000000000 l O .bss^^{{.*}}lto-linker-scripts-thinlto.ll.tmp 0000000000000004 g
; CHECK: 0000000000000000 g F .text^^{{.*}}lto-linker-scripts-thinlto.ll.tmp 0000000000000008 foo

; IR: @g = internal global i32 0, section ".bss^^{{.*}}", !guid !0 #0
; IR: define ptr @foo() #1 section ".text^^{{.*}}" !guid !1 {
; IR: attributes #0 = { "linker_output_section"="bar" }
; IR: attributes #1 = { "linker_output_section"="foo" }

; INDEX: sectionInfo: (outputSectionName: "foo"))
; INDEX: sectionInfo: (outputSectionName: "bar"))

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@g = internal global i32 0

define ptr @foo(){
  ret ptr @g
}
