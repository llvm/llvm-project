; RUN: opt -mtriple=x86_64-unknown-linux-gnu -S -debugify -codegenprepare < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

declare void @foo(i32)

declare void @slowpath(i32, i32*)

; Is DI maintained after sinking bitcast?
define void @test(i1 %cond, i64* %base) {
; CHECK-LABEL: @test
entry:
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK: bitcast i64* %addr to i32*, !dbg ![[castLoc:[0-9]+]]
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
  %cmp = icmp eq i32 %v1, 0
  br i1 %cmp, label %rare.1, label %fallthrough

fallthrough:
  ret void

rare.1:
; CHECK-LABEL: rare.1:
; CHECK: bitcast i64* %addr to i32*, !dbg ![[castLoc]]
  call void @slowpath(i32 %v1, i32* %casted) ;; NOT COLD
  br label %fallthrough
}

; CHECK: ![[castLoc]] = !DILocation(line: 2
