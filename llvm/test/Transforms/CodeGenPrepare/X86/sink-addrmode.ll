; RUN: opt -S -codegenprepare < %s | FileCheck %s

target datalayout =
"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-unknown-linux-gnu"

; Can we sink single addressing mode computation to use?
define void @test1(i1 %cond, i64* %base) {
; CHECK-LABEL: @test1
; CHECK: add i64 {{.+}}, 40
entry:
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
  %v = load i32, i32* %casted, align 4
  br label %fallthrough

fallthrough:
  ret void
}

declare void @foo(i32)

; Make sure sinking two copies of addressing mode into different blocks works
define void @test2(i1 %cond, i64* %base) {
; CHECK-LABEL: @test2
entry:
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK: add i64 {{.+}}, 40
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
  %cmp = icmp eq i32 %v1, 0
  br i1 %cmp, label %next, label %fallthrough

next:
; CHECK-LABEL: next:
; CHECK: add i64 {{.+}}, 40
  %v2 = load i32, i32* %casted, align 4
  call void @foo(i32 %v2)
  br label %fallthrough

fallthrough:
  ret void
}

; If we have two loads in the same block, only need one copy of addressing mode
; - instruction selection will duplicate if needed
define void @test3(i1 %cond, i64* %base) {
; CHECK-LABEL: @test3
entry:
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK: add i64 {{.+}}, 40
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
; CHECK-NOT: add i64 {{.+}}, 40
  %v2 = load i32, i32* %casted, align 4
  call void @foo(i32 %v2)
  br label %fallthrough

fallthrough:
  ret void
}

; Can we still sink addressing mode if there's a cold use of the
; address itself?  
define void @test4(i1 %cond, i64* %base) {
; CHECK-LABEL: @test4
entry:
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK: add i64 {{.+}}, 40
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
  %cmp = icmp eq i32 %v1, 0
  br i1 %cmp, label %rare.1, label %fallthrough

fallthrough:
  ret void

rare.1:
; CHECK-LABEL: rare.1:
; CHECK: add i64 {{.+}}, 40
  call void @slowpath(i32 %v1, i32* %casted) cold
  br label %fallthrough
}

; Negative test - don't want to duplicate addressing into hot path
define void @test5(i1 %cond, i64* %base) {
; CHECK-LABEL: @test5
entry:
; CHECK: %addr = getelementptr
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK-NOT: add i64 {{.+}}, 40
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
  %cmp = icmp eq i32 %v1, 0
  br i1 %cmp, label %rare.1, label %fallthrough

fallthrough:
  ret void

rare.1:
  call void @slowpath(i32 %v1, i32* %casted) ;; NOT COLD
  br label %fallthrough
}

; Negative test - opt for size
define void @test6(i1 %cond, i64* %base) minsize {
; CHECK-LABEL: @test6
entry:
; CHECK: %addr = getelementptr
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK-NOT: add i64 {{.+}}, 40
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
  %cmp = icmp eq i32 %v1, 0
  br i1 %cmp, label %rare.1, label %fallthrough

fallthrough:
  ret void

rare.1:
  call void @slowpath(i32 %v1, i32* %casted) cold
  br label %fallthrough
}


; Make sure sinking two copies of addressing mode into different blocks works
; when there are cold paths for each.
define void @test7(i1 %cond, i64* %base) {
; CHECK-LABEL: @test7
entry:
  %addr = getelementptr inbounds i64, i64* %base, i64 5
  %casted = bitcast i64* %addr to i32*
  br i1 %cond, label %if.then, label %fallthrough

if.then:
; CHECK-LABEL: if.then:
; CHECK: add i64 {{.+}}, 40
  %v1 = load i32, i32* %casted, align 4
  call void @foo(i32 %v1)
  %cmp = icmp eq i32 %v1, 0
  br i1 %cmp, label %rare.1, label %next

next:
; CHECK-LABEL: next:
; CHECK: add i64 {{.+}}, 40
  %v2 = load i32, i32* %casted, align 4
  call void @foo(i32 %v2)
  %cmp2 = icmp eq i32 %v2, 0
  br i1 %cmp2, label %rare.1, label %fallthrough

fallthrough:
  ret void

rare.1:
; CHECK-LABEL: rare.1:
; CHECK: add i64 {{.+}}, 40
  call void @slowpath(i32 %v1, i32* %casted) cold
  br label %next

rare.2:
; CHECK-LABEL: rare.2:
; CHECK: add i64 {{.+}}, 40
  call void @slowpath(i32 %v2, i32* %casted) cold
  br label %fallthrough
}


declare void @slowpath(i32, i32*)
