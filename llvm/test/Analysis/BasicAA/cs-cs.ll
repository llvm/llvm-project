; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "arm-apple-ios"

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1) #0
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i1) #0
declare void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr nocapture writeonly, ptr nocapture readonly, i64, i32) #0

declare void @a_readonly_func(ptr) #1
declare void @a_writeonly_func(ptr) #2

define void @test2(ptr %P, ptr %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test2:

; CHECK:   MayAlias:     i8* %P, i8* %Q
; CHECK:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK:   Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK:   Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test2_atomic(ptr %P, ptr %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
  ret void

; CHECK-LABEL: Function: test2_atomic:

; CHECK:   MayAlias:     i8* %P, i8* %Q
; CHECK:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; CHECK:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; CHECK:   Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; CHECK:   Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; CHECK:   Both ModRef:   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1) <->   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
; CHECK:   Both ModRef:   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1) <->   tail call void @llvm.memcpy.element.unordered.atomic.p0.p0.i64(ptr align 1 %P, ptr align 1 %Q, i64 12, i32 1)
}

define void @test2a(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test2a:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test2b(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 12
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test2b:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test2c(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 11
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test2c:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test2d(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 -12
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test2d:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test2e(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  %R = getelementptr i8, ptr %P, i64 -11
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test2e:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %R, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test3(ptr %P, ptr %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test3:

; CHECK: MayAlias:     i8* %P, i8* %Q
; CHECK: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
}

define void @test3a(ptr noalias %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test3a:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 8, i1 false)
}

define void @test4(ptr %P, ptr noalias %Q) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test4:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
; CHECK: NoModRef:  Ptr: i8* %Q        <->  tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memset.p0.i64(ptr %P, i8 42, i64 8, i1 false)
}

define void @test5(ptr %P, ptr %Q, ptr %R) #3 {
  load i8, ptr %P
  load i8, ptr %Q
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test5:

; CHECK: MayAlias:     i8* %P, i8* %Q
; CHECK: MayAlias:     i8* %P, i8* %R
; CHECK: MayAlias:     i8* %Q, i8* %R
; CHECK: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Both ModRef:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test5a(ptr noalias %P, ptr noalias %Q, ptr noalias %R) nounwind ssp {
  load i8, ptr %P
  load i8, ptr %Q
  load i8, ptr %R
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
  ret void

; CHECK-LABEL: Function: test5a:

; CHECK: NoAlias:     i8* %P, i8* %Q
; CHECK: NoAlias:     i8* %P, i8* %R
; CHECK: NoAlias:     i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: NoModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Just Ref:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %R, i64 12, i1 false) <->   tail call void @llvm.memcpy.p0.p0.i64(ptr %P, ptr %Q, i64 12, i1 false)
}

define void @test6(ptr %P) #3 {
  load i8, ptr %P
  call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false)
  call void @a_readonly_func(ptr %P)
  ret void

; CHECK-LABEL: Function: test6:

; CHECK: Just Mod:  Ptr: i8* %P        <->  call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false)
; CHECK: Just Ref:  Ptr: i8* %P        <->  call void @a_readonly_func(ptr %P)
; CHECK: Just Mod:   call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false) <->   call void @a_readonly_func(ptr %P)
; CHECK: Just Ref:   call void @a_readonly_func(ptr %P) <->   call void @llvm.memset.p0.i64(ptr align 8 %P, i8 -51, i64 32, i1 false)
}

define void @test7(ptr %P) #3 {
  load i8, ptr %P
  call void @a_writeonly_func(ptr %P)
  call void @a_readonly_func(ptr %P)
  ret void

; CHECK-LABEL: Function: test7:

; CHECK: Just Mod:  Ptr: i8* %P        <->  call void @a_writeonly_func(ptr %P)
; CHECK: Just Ref:  Ptr: i8* %P        <->  call void @a_readonly_func(ptr %P)
; CHECK: Just Mod:   call void @a_writeonly_func(ptr %P) <->   call void @a_readonly_func(ptr %P)
; CHECK: Just Ref:   call void @a_readonly_func(ptr %P) <->   call void @a_writeonly_func(ptr %P)
}

declare void @an_inaccessiblememonly_func() #4
declare void @an_inaccessibleorargmemonly_func(ptr) #5
declare void @an_argmemonly_func(ptr) #0

define void @test8(ptr %p) {
entry:
  %q = getelementptr i8, ptr %p, i64 16
  load i8, ptr %p
  load i8, ptr %q
  call void @a_readonly_func(ptr %p)
  call void @an_inaccessiblememonly_func()
  call void @a_writeonly_func(ptr %q)
  call void @an_inaccessiblememonly_func()
  call void @an_inaccessibleorargmemonly_func(ptr %q)
  call void @an_argmemonly_func(ptr %q)
  ret void

; CHECK-LABEL: Function: test8
; CHECK: NoModRef:  Ptr: i8* %p <->  call void @an_inaccessiblememonly_func()
; CHECK: NoModRef:  Ptr: i8* %q <->  call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef:  Ptr: i8* %p <->  call void @an_inaccessibleorargmemonly_func(ptr %q)
; CHECK: Both ModRef:  Ptr: i8* %q <->  call void @an_inaccessibleorargmemonly_func(ptr %q)
; CHECK: Both ModRef:  Ptr: i8* %p <->  call void @an_argmemonly_func(ptr %q)
; CHECK: Both ModRef:  Ptr: i8* %q <->  call void @an_argmemonly_func(ptr %q)
; CHECK: Just Ref: call void @a_readonly_func(ptr %p) <-> call void @an_inaccessiblememonly_func()
; CHECK: Just Ref: call void @a_readonly_func(ptr %p) <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
; CHECK: Just Ref: call void @a_readonly_func(ptr %p) <-> call void @an_argmemonly_func(ptr %q)
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @a_readonly_func(ptr %p)
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @a_writeonly_func(ptr %q)
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
; CHECK: NoModRef: call void @an_inaccessiblememonly_func() <-> call void @an_argmemonly_func(ptr %q)
; CHECK: Just Mod: call void @a_writeonly_func(ptr %q) <-> call void @an_inaccessiblememonly_func()
; CHECK: Just Mod: call void @a_writeonly_func(ptr %q) <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
; CHECK: Just Mod: call void @a_writeonly_func(ptr %q) <-> call void @an_argmemonly_func(ptr %q)
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @a_readonly_func(ptr %p)
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @a_writeonly_func(ptr %q)
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(ptr %q) <-> call void @an_argmemonly_func(ptr %q)
; CHECK: Both ModRef: call void @an_argmemonly_func(ptr %q) <-> call void @a_readonly_func(ptr %p)
; CHECK: Both ModRef: call void @an_argmemonly_func(ptr %q) <-> call void @a_writeonly_func(ptr %q)
; CHECK: NoModRef: call void @an_argmemonly_func(ptr %q) <-> call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef: call void @an_argmemonly_func(ptr %q) <-> call void @an_inaccessibleorargmemonly_func(ptr %q)
}

;; test that MustAlias is set for calls when no MayAlias is found.
declare void @another_argmemonly_func(ptr, ptr) #0
define void @test8a(ptr noalias %p, ptr noalias %q) {
entry:
  load i8, ptr %p
  load i8, ptr %q
  call void @another_argmemonly_func(ptr %p, ptr %q)
  ret void

; CHECK-LABEL: Function: test8a
; CHECK: Both ModRef:  Ptr: i8* %p <->  call void @another_argmemonly_func(ptr %p, ptr %q)
; CHECK: Both ModRef:  Ptr: i8* %q <->  call void @another_argmemonly_func(ptr %p, ptr %q)
}
define void @test8b(ptr %p, ptr %q) {
entry:
  load i8, ptr %p
  load i8, ptr %q
  call void @another_argmemonly_func(ptr %p, ptr %q)
  ret void

; CHECK-LABEL: Function: test8b
; CHECK: Both ModRef:  Ptr: i8* %p <->  call void @another_argmemonly_func(ptr %p, ptr %q)
; CHECK: Both ModRef:  Ptr: i8* %q <->  call void @another_argmemonly_func(ptr %p, ptr %q)
}


;; test that unknown operand bundle has unknown effect to the heap
define void @test9(ptr %p) {
; CHECK-LABEL: Function: test9
entry:
  %q = getelementptr i8, ptr %p, i64 16
  load i8, ptr %p
  load i8, ptr %q
  call void @a_readonly_func(ptr %p) [ "unknown"() ]
  call void @an_inaccessiblememonly_func() [ "unknown"() ]
  call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
  call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
  ret void

; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @a_readonly_func(ptr %p) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @a_readonly_func(ptr %p) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @a_readonly_func(ptr %p) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:   call void @a_readonly_func(ptr %p) [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @a_readonly_func(ptr %p) [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(ptr %q) [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(ptr %q) [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) [ "unknown"() ]
}

;; test callsite overwrite of unknown operand bundle
define void @test10(ptr %p) {
; CHECK-LABEL: Function: test10
entry:
  %q = getelementptr i8, ptr %p, i64 16
  load i8, ptr %p
  load i8, ptr %q
  call void @a_readonly_func(ptr %p) #6 [ "unknown"() ]
  call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
  call void @an_inaccessibleorargmemonly_func(ptr %q) #8 [ "unknown"() ]
  call void @an_argmemonly_func(ptr %q) #9 [ "unknown"() ]
  ret void

; CHECK: Just Ref:  Ptr: i8* %p        <->  call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; CHECK: Just Ref:  Ptr: i8* %q        <->  call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; CHECK: NoModRef:  Ptr: i8* %p        <->  call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; CHECK: NoModRef:  Ptr: i8* %q        <->  call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p        <->  call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p        <->  call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; CHECK: Just Ref:   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; CHECK: Just Ref:   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; CHECK: Just Ref:   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
; CHECK: NoModRef:   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ] <->   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ] <->   call void @a_readonly_func(ptr %p) #9 [ "unknown"() ]
; CHECK: NoModRef:   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #10 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(ptr %q) #12 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(ptr %q) #11 [ "unknown"() ]
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { noinline nounwind readonly }
attributes #2 = { noinline nounwind writeonly }
attributes #3 = { nounwind ssp }
attributes #4 = { inaccessiblememonly nounwind }
attributes #5 = { inaccessiblemem_or_argmemonly nounwind }
attributes #6 = { readonly }
attributes #7 = { inaccessiblememonly }
attributes #8 = { inaccessiblemem_or_argmemonly }
attributes #9 = { argmemonly }
