; RUN: llc -verify-machineinstrs -O3 -mtriple=x86_64-unknown-unknown -mcpu=skylake -filetype=obj < %s | llvm-objdump -d --no-show-raw-insn - | FileCheck %s

; TODO: At the moment, autopadding for SKX102 is not the default, but
; eventually we'd like ti to be for the integrated assembler (only).

target datalayout = "e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

define void @test(i1 %c) {
; CHECK: 0: pushq
; CHECK-NEXT: 1: movl
; CHECK-NEXT: 3: callq
; CHECK-NEXT: 8: callq
; CHECK-NEXT: d: callq
; CHECK-NEXT: 12: callq
; CHECK-NEXT: 17: callq
; TODO: want a nop here
; CHECK-NEXT: 1c: testb
; CHECK-NEXT: 1f: je
entry:
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  call void @foo()
  br i1 %c, label %taken, label %untaken

taken:
  call void @foo()
  ret void
untaken:
  call void @bar()
  ret void
}

declare void @foo()
declare void @bar()
