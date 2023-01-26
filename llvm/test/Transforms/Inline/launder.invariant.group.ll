; RUN: opt -S -passes=inline < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck %s
; RUN: opt -S -passes=inline -inline-threshold=1 < %s | FileCheck %s

%struct.A = type <{ ptr, i32, [4 x i8] }>

; This test checks if value returned from the launder is considered aliasing
; with its argument.  Due to bug caused by handling launder in capture tracking
; sometimes it would be considered noalias.
; CHECK-LABEL: define i32 @bar(ptr noalias
define i32 @bar(ptr noalias) {
; CHECK-NOT: noalias
  %2 = call ptr @llvm.launder.invariant.group.p0(ptr %0)
  %3 = getelementptr inbounds i8, ptr %2, i64 8
  store i32 42, ptr %3, align 8
  %4 = getelementptr inbounds %struct.A, ptr %0, i64 0, i32 1
  %5 = load i32, ptr %4, align 8
  ret i32 %5
}

; CHECK-LABEL: define i32 @foo(ptr noalias
define i32 @foo(ptr noalias)  {
  ; CHECK-NOT: call i32 @bar(
  ; CHECK-NOT: !noalias
  %2 = tail call i32 @bar(ptr %0)
  ret i32 %2
}


; This test checks if invariant group intrinsics have zero cost for inlining.
; CHECK-LABEL: define ptr @caller(ptr
define ptr @caller(ptr %p) {
; CHECK-NOT: call ptr @lot_of_launders_and_strips
  %a1 = call ptr @lot_of_launders_and_strips(ptr %p)
  %a2 = call ptr @lot_of_launders_and_strips(ptr %a1)
  %a3 = call ptr @lot_of_launders_and_strips(ptr %a2)
  %a4 = call ptr @lot_of_launders_and_strips(ptr %a3)
  ret ptr %a4
}

define ptr @lot_of_launders_and_strips(ptr %p) {
  %a1 = call ptr @llvm.launder.invariant.group.p0(ptr %p)
  %a2 = call ptr @llvm.launder.invariant.group.p0(ptr %a1)
  %a3 = call ptr @llvm.launder.invariant.group.p0(ptr %a2)
  %a4 = call ptr @llvm.launder.invariant.group.p0(ptr %a3)

  %s1 = call ptr @llvm.strip.invariant.group.p0(ptr %a4)
  %s2 = call ptr @llvm.strip.invariant.group.p0(ptr %s1)
  %s3 = call ptr @llvm.strip.invariant.group.p0(ptr %s2)
  %s4 = call ptr @llvm.strip.invariant.group.p0(ptr %s3)

   ret ptr %s4
}


declare ptr @llvm.launder.invariant.group.p0(ptr)
declare ptr @llvm.strip.invariant.group.p0(ptr)
