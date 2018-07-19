; RUN: opt -S -gvn < %s | FileCheck %s
; RUN: opt -S -newgvn < %s | FileCheck %s
; RUN: opt -S -O3 < %s | FileCheck %s

; This test check if optimizer is not proving equality based on mustalias
; CHECK-LABEL: define void @dontProveEquality(i8* %a) 
define void @dontProveEquality(i8* %a) {
  %b = call i8* @llvm.invariant.group.barrier(i8* %a)
  %r = i1 icmp eq i8* %b, i8* %a
;CHECK: call void @use(%r)
  call void @use(%r)
}

declare void @use(i1)
declare i8* @llvm.invariant.group.barrier(i8 *)
