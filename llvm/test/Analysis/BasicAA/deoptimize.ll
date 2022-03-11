; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

@G1 = external global i32

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)
declare void @llvm.memmove.p0i8.p0i8.i64(i8* nocapture, i8* nocapture, i64, i1)
declare void @llvm.experimental.deoptimize.void(...)
declare void @unknown_but_readonly() readonly

define void @test1(i8* %p) {
  call void(...) @llvm.experimental.deoptimize.void() [ "deopt"() ]
  ret void

; CHECK-LABEL: Function: test1:
; CHECK:  Just Ref: Ptr: i8* %p <-> call void (...) @llvm.experimental.deoptimize.isVoid() [ "deopt"() ]
}

; By specification calls with deopt bundles reads through all operands and entire heap.
; Check that global G1 is reported as Ref by memcpy/memmove calls.
define i32 @test_memcpy_with_deopt() {
; CHECK-LABEL: Function: test_memcpy_with_deopt:
; CHECK: Just Mod:  Ptr: i8* %A	<->  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]
; CHECK: Just Ref:  Ptr: i8* %B	<->  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]
; CHECK: Just Ref:  Ptr: i32* @G1	<->  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]

  %A = alloca i8
  %B = alloca i8

  store i32 2, i32* @G1  ;; Not referenced by semantics of memcpy but still may be read due to "deopt"

  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]

  %C = load i32, i32* @G1
  ret i32 %C
}

define i32 @test_memmove_with_deopt() {
; CHECK-LABEL: Function: test_memmove_with_deopt:
; CHECK: Just Mod:  Ptr: i8* %A	<->  call void @llvm.memmove.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]
; CHECK: Just Ref:  Ptr: i8* %B	<->  call void @llvm.memmove.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]
; CHECK: Just Ref:  Ptr: i32* @G1	<->  call void @llvm.memmove.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]

  %A = alloca i8
  %B = alloca i8

  store i32 2, i32* @G1  ;; Not referenced by semantics of memcpy but still may be read due to "deopt"

  call void @llvm.memmove.p0i8.p0i8.i64(i8* %A, i8* %B, i64 -1, i1 false) [ "deopt"() ]

  %C = load i32, i32* @G1
  ret i32 %C
}
