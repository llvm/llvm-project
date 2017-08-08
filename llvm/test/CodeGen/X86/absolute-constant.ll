; RUN: llc < %s | FileCheck %s
; RUN: llc -relocation-model=pic < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@foo = external global i8, align 1, !absolute_symbol !0

define void @bar(i8* %x) {
entry:
  %0 = load i8, i8* %x, align 1
  %conv = sext i8 %0 to i32
  ; CHECK: testb $foo, (%rdi)
  %and = and i32 %conv, sext (i8 ptrtoint (i8* @foo to i8) to i32)
  %tobool = icmp eq i32 %and, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  tail call void (...) @xf()
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  ret void
}

declare void @xf(...)

!0 = !{i32 0, i32 256}
