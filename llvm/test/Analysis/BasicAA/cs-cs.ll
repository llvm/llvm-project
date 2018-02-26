; RUN: opt < %s -basicaa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "arm-apple-ios"

declare void @llvm.memset.p0i8.i64(i8* nocapture writeonly, i8, i64, i32, i1) #0
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1) #0

declare void @a_readonly_func(i8*) #1
declare void @a_writeonly_func(i8*) #2

define void @test2(i8* %P, i8* %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test2:

; CHECK:   MayAlias:     i8* %P, i8* %Q
; CHECK:   Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK:   Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK:   Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK:   Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK:   Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK:   Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test2a(i8* noalias %P, i8* noalias %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test2a:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test2b(i8* noalias %P, i8* noalias %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  %R = getelementptr i8, i8* %P, i64 12
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test2b:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test2c(i8* noalias %P, i8* noalias %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  %R = getelementptr i8, i8* %P, i64 11
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test2c:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test2d(i8* noalias %P, i8* noalias %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  %R = getelementptr i8, i8* %P, i64 -12
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test2d:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test2e(i8* noalias %P, i8* noalias %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  %R = getelementptr i8, i8* %P, i64 -11
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test2e:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: NoAlias:      i8* %P, i8* %R
; CHECK: NoAlias:      i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %R        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %R, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test3(i8* %P, i8* %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test3:

; CHECK: MayAlias:     i8* %P, i8* %Q
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
}

define void @test3a(i8* noalias %P, i8* noalias %Q) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test3a:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 8, i32 1, i1 false)
}

define void @test4(i8* %P, i8* noalias %Q) #3 {
  tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test4:

; CHECK: NoAlias:      i8* %P, i8* %Q
; CHECK: Just Mod (MustAlias):  Ptr: i8* %P        <->  tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %Q        <->  tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q        <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memset.p0i8.i64(i8* %P, i8 42, i64 8, i32 1, i1 false)
}

define void @test5(i8* %P, i8* %Q, i8* %R) #3 {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test5:

; CHECK: MayAlias:     i8* %P, i8* %Q
; CHECK: MayAlias:     i8* %P, i8* %R
; CHECK: MayAlias:     i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Both ModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test5a(i8* noalias %P, i8* noalias %Q, i8* noalias %R) nounwind ssp {
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
  ret void

; CHECK-LABEL: Function: test5a:

; CHECK: NoAlias:     i8* %P, i8* %Q
; CHECK: NoAlias:     i8* %P, i8* %R
; CHECK: NoAlias:     i8* %Q, i8* %R
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
; CHECK: Just Mod:  Ptr: i8* %P     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: NoModRef:  Ptr: i8* %Q     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Just Ref:  Ptr: i8* %R     <->  tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false)
; CHECK: Just Mod:   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %R, i64 12, i32 1, i1 false) <->   tail call void @llvm.memcpy.p0i8.p0i8.i64(i8* %P, i8* %Q, i64 12, i32 1, i1 false)
}

define void @test6(i8* %P) #3 {
  call void @llvm.memset.p0i8.i64(i8* %P, i8 -51, i64 32, i32 8, i1 false)
  call void @a_readonly_func(i8* %P)
  ret void

; CHECK-LABEL: Function: test6:

; CHECK: Just Mod (MustAlias):  Ptr: i8* %P        <->  call void @llvm.memset.p0i8.i64(i8* %P, i8 -51, i64 32, i32 8, i1 false)
; CHECK: Just Ref:  Ptr: i8* %P        <->  call void @a_readonly_func(i8* %P)
; CHECK: Just Mod:   call void @llvm.memset.p0i8.i64(i8* %P, i8 -51, i64 32, i32 8, i1 false) <->   call void @a_readonly_func(i8* %P)
; CHECK: Just Ref:   call void @a_readonly_func(i8* %P) <->   call void @llvm.memset.p0i8.i64(i8* %P, i8 -51, i64 32, i32 8, i1 false)
}

define void @test7(i8* %P) #3 {
  call void @a_writeonly_func(i8* %P)
  call void @a_readonly_func(i8* %P)
  ret void

; CHECK-LABEL: Function: test7:

; CHECK: Just Mod:  Ptr: i8* %P        <->  call void @a_writeonly_func(i8* %P)
; CHECK: Just Ref:  Ptr: i8* %P        <->  call void @a_readonly_func(i8* %P)
; CHECK: Just Mod:   call void @a_writeonly_func(i8* %P) <->   call void @a_readonly_func(i8* %P)
; CHECK: Just Ref:   call void @a_readonly_func(i8* %P) <->   call void @a_writeonly_func(i8* %P)
}

declare void @an_inaccessiblememonly_func() #4
declare void @an_inaccessibleorargmemonly_func(i8*) #5
declare void @an_argmemonly_func(i8*) #0

define void @test8(i8* %p) {
entry:
  %q = getelementptr i8, i8* %p, i64 16
  call void @a_readonly_func(i8* %p)
  call void @an_inaccessiblememonly_func()
  call void @a_writeonly_func(i8* %q)
  call void @an_inaccessiblememonly_func()
  call void @an_inaccessibleorargmemonly_func(i8* %q)
  call void @an_argmemonly_func(i8* %q)
  ret void

; CHECK-LABEL: Function: test8
; CHECK: NoModRef:  Ptr: i8* %p <->  call void @an_inaccessiblememonly_func()
; CHECK: NoModRef:  Ptr: i8* %q <->  call void @an_inaccessiblememonly_func()
; CHECK: NoModRef:  Ptr: i8* %p <->  call void @an_inaccessibleorargmemonly_func(i8* %q)
; CHECK: Both ModRef (MustAlias):  Ptr: i8* %q <->  call void @an_inaccessibleorargmemonly_func(i8* %q)
; CHECK: NoModRef:  Ptr: i8* %p <->  call void @an_argmemonly_func(i8* %q)
; CHECK: Both ModRef (MustAlias):  Ptr: i8* %q <->  call void @an_argmemonly_func(i8* %q)
; CHECK: Just Ref: call void @a_readonly_func(i8* %p) <-> call void @an_inaccessiblememonly_func()
; CHECK: Just Ref: call void @a_readonly_func(i8* %p) <-> call void @an_inaccessibleorargmemonly_func(i8* %q)
; CHECK: Just Ref: call void @a_readonly_func(i8* %p) <-> call void @an_argmemonly_func(i8* %q)
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @a_readonly_func(i8* %p)
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @a_writeonly_func(i8* %q)
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef: call void @an_inaccessiblememonly_func() <-> call void @an_inaccessibleorargmemonly_func(i8* %q)
; CHECK: NoModRef: call void @an_inaccessiblememonly_func() <-> call void @an_argmemonly_func(i8* %q)
; CHECK: Just Mod: call void @a_writeonly_func(i8* %q) <-> call void @an_inaccessiblememonly_func()
; CHECK: Just Mod: call void @a_writeonly_func(i8* %q) <-> call void @an_inaccessibleorargmemonly_func(i8* %q)
; CHECK: Just Mod: call void @a_writeonly_func(i8* %q) <-> call void @an_argmemonly_func(i8* %q)
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(i8* %q) <-> call void @a_readonly_func(i8* %p)
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(i8* %q) <-> call void @a_writeonly_func(i8* %q)
; CHECK: Both ModRef: call void @an_inaccessibleorargmemonly_func(i8* %q) <-> call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef (MustAlias): call void @an_inaccessibleorargmemonly_func(i8* %q) <-> call void @an_argmemonly_func(i8* %q)
; CHECK: Both ModRef: call void @an_argmemonly_func(i8* %q) <-> call void @a_readonly_func(i8* %p)
; CHECK: Both ModRef: call void @an_argmemonly_func(i8* %q) <-> call void @a_writeonly_func(i8* %q)
; CHECK: NoModRef: call void @an_argmemonly_func(i8* %q) <-> call void @an_inaccessiblememonly_func()
; CHECK: Both ModRef (MustAlias): call void @an_argmemonly_func(i8* %q) <-> call void @an_inaccessibleorargmemonly_func(i8* %q)
}

;; test that MustAlias is set for calls when no MayAlias is found.
declare void @another_argmemonly_func(i8*, i8*) #0
define void @test8a(i8* noalias %p, i8* noalias %q) {
entry:
  call void @another_argmemonly_func(i8* %p, i8* %q)
  ret void

; CHECK-LABEL: Function: test8a
; CHECK: Both ModRef:  Ptr: i8* %p <->  call void @another_argmemonly_func(i8* %p, i8* %q)
; CHECK: Both ModRef:  Ptr: i8* %q <->  call void @another_argmemonly_func(i8* %p, i8* %q)
}
define void @test8b(i8* %p, i8* %q) {
entry:
  call void @another_argmemonly_func(i8* %p, i8* %q)
  ret void

; CHECK-LABEL: Function: test8b
; CHECK: Both ModRef:  Ptr: i8* %p <->  call void @another_argmemonly_func(i8* %p, i8* %q)
; CHECK: Both ModRef:  Ptr: i8* %q <->  call void @another_argmemonly_func(i8* %p, i8* %q)
}


;; test that unknown operand bundle has unknown effect to the heap
define void @test9(i8* %p) {
; CHECK-LABEL: Function: test9
entry:
  %q = getelementptr i8, i8* %p, i64 16
  call void @a_readonly_func(i8* %p) [ "unknown"() ]
  call void @an_inaccessiblememonly_func() [ "unknown"() ]
  call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ]
  call void @an_argmemonly_func(i8* %q) [ "unknown"() ]
  ret void

; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @a_readonly_func(i8* %p) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @a_readonly_func(i8* %p) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %p     <->  call void @an_argmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @an_argmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @a_readonly_func(i8* %p) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:   call void @a_readonly_func(i8* %p) [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @a_readonly_func(i8* %p) [ "unknown"() ] <->   call void @an_argmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @a_readonly_func(i8* %p) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() [ "unknown"() ] <->   call void @an_argmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ] <->   call void @a_readonly_func(i8* %p) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ] <->   call void @an_argmemonly_func(i8* %q) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(i8* %q) [ "unknown"() ] <->   call void @a_readonly_func(i8* %p) [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(i8* %q) [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(i8* %q) [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(i8* %q) [ "unknown"() ]
}

;; test callsite overwrite of unknown operand bundle
define void @test10(i8* %p) {
; CHECK-LABEL: Function: test10
entry:
  %q = getelementptr i8, i8* %p, i64 16
  call void @a_readonly_func(i8* %p) #6 [ "unknown"() ]
  call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
  call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ]
  call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ]
  ret void

; CHECK: Just Ref:  Ptr: i8* %p        <->  call void @a_readonly_func(i8* %p) #6 [ "unknown"() ]
; CHECK: Just Ref:  Ptr: i8* %q        <->  call void @a_readonly_func(i8* %p) #6 [ "unknown"() ]
; CHECK: NoModRef:  Ptr: i8* %p        <->  call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
; CHECK: NoModRef:  Ptr: i8* %q        <->  call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
; CHECK: NoModRef:  Ptr: i8* %p        <->  call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ]
; CHECK: Both ModRef (MustAlias):  Ptr: i8* %q     <->  call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ]
; CHECK: NoModRef:  Ptr: i8* %p        <->  call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ]
; CHECK: Both ModRef (MustAlias):  Ptr: i8* %q     <->  call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ]
; CHECK: Just Ref:   call void @a_readonly_func(i8* %p) #6 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
; CHECK: Just Ref:   call void @a_readonly_func(i8* %p) #6 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ]
; CHECK: Just Ref:   call void @a_readonly_func(i8* %p) #6 [ "unknown"() ] <->   call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() #7 [ "unknown"() ] <->   call void @a_readonly_func(i8* %p) #6 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessiblememonly_func() #7 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ]
; CHECK: NoModRef:   call void @an_inaccessiblememonly_func() #7 [ "unknown"() ] <->   call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ] <->   call void @a_readonly_func(i8* %p) #6 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
; CHECK: Both ModRef (MustAlias):   call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ] <->   call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ]
; CHECK: Both ModRef:   call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ] <->   call void @a_readonly_func(i8* %p) #6 [ "unknown"() ]
; CHECK: NoModRef:   call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ] <->   call void @an_inaccessiblememonly_func() #7 [ "unknown"() ]
; CHECK: Both ModRef (MustAlias):   call void @an_argmemonly_func(i8* %q) #9 [ "unknown"() ] <->   call void @an_inaccessibleorargmemonly_func(i8* %q) #8 [ "unknown"() ]
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
