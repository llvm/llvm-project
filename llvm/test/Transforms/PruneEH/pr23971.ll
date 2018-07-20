; RUN: opt -S -prune-eh < %s | FileCheck %s
; RUN: opt -S -passes='function-attrs,function(simplify-cfg)' < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @f() #0 {
entry:
  call void asm sideeffect "ret\0A\09", "~{dirflag},~{fpsr},~{flags}"()
  unreachable
}

define i32 @g() {
entry:
  call void @f()
  ret i32 42
}

; CHECK-LABEL: define i32 @g()
; CHECK: ret i32 42

attributes #0 = { naked noinline }
