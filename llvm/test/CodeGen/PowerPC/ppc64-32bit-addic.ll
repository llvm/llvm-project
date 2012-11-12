; Check that the ADDIC optimizations are not applied on PPC64
; RUN: llc < %s | FileCheck %s
; ModuleID = 'os_unix.c'
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-freebsd9.0"

define i32 @notZero(i32 %call) nounwind {
entry:
; CHECK-NOT: addic
  %not.tobool = icmp ne i32 %call, 0
  %. = zext i1 %not.tobool to i32
  ret i32 %.
}

define i32 @isMinusOne(i32 %call) nounwind {
entry:
; CHECK-NOT: addic
  %not.tobool = icmp eq i32 %call, -1
  %. = zext i1 %not.tobool to i32
  ret i32 %.
}

define i32 @isNotMinusOne(i32 %call) nounwind {
entry:
; CHECK-NOT: addic
  %not.tobool = icmp ne i32 %call, -1
  %. = zext i1 %not.tobool to i32
  ret i32 %.
}
