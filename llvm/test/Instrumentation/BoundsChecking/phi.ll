; RUN: opt < %s -bounds-checking -S | FileCheck %s
target datalayout = "e-p:64:64:64-p1:16:16:16-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

@global = private unnamed_addr constant [10 x i8] c"ola\00mundo\00", align 1

; CHECK: f1
; no checks are possible here
; CHECK-NOT: trap
define void @f1(i8* nocapture %c) {
entry:
  %0 = load i8, i8* %c, align 1
  %tobool1 = icmp eq i8 %0, 0
  br i1 %tobool1, label %while.end, label %while.body

while.body:
  %c.addr.02 = phi i8* [ %incdec.ptr, %while.body ], [ %c, %entry ]
  %incdec.ptr = getelementptr inbounds i8, i8* %c.addr.02, i64 -1
  store i8 100, i8* %c.addr.02, align 1
  %1 = load i8, i8* %incdec.ptr, align 1
  %tobool = icmp eq i8 %1, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}


; CHECK: f2
define void @f2() {
while.body.i.preheader:
  %addr = getelementptr inbounds [10 x i8], [10 x i8]* @global, i64 0, i64 9
  br label %while.body.i

while.body.i:
; CHECK: phi
; CHECK-NEXT: phi
; CHECK-NOT: phi
  %c.addr.02.i = phi i8* [ %incdec.ptr.i, %while.body.i ], [ %addr, %while.body.i.preheader ]
  %incdec.ptr.i = getelementptr inbounds i8, i8* %c.addr.02.i, i64 -1
; CHECK: sub i64 10, %0
; CHECK-NEXT: icmp ult i64 10, %0
; CHECK-NEXT: icmp ult i64 {{.*}}, 1
; CHECK-NEXT: or i1
; CHECK-NEXT: br {{.*}}, label %trap
  store i8 100, i8* %c.addr.02.i, align 1
  %0 = load i8, i8* %incdec.ptr.i, align 1
  %tobool.i = icmp eq i8 %0, 0
  br i1 %tobool.i, label %fn.exit, label %while.body.i

fn.exit:
  ret void
}


@global_as1 = private unnamed_addr addrspace(1) constant [10 x i8] c"ola\00mundo\00", align 1

define void @f1_as1(i8 addrspace(1)* nocapture %c) {
; CHECK: @f1_as1
; no checks are possible here
; CHECK-NOT: trap
entry:
  %0 = load i8, i8 addrspace(1)* %c, align 1
  %tobool1 = icmp eq i8 %0, 0
  br i1 %tobool1, label %while.end, label %while.body

while.body:
  %c.addr.02 = phi i8 addrspace(1)* [ %incdec.ptr, %while.body ], [ %c, %entry ]
  %incdec.ptr = getelementptr inbounds i8, i8 addrspace(1)* %c.addr.02, i64 -1
  store i8 100, i8 addrspace(1)* %c.addr.02, align 1
  %1 = load i8, i8 addrspace(1)* %incdec.ptr, align 1
  %tobool = icmp eq i8 %1, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}


define void @f2_as1() {
; CHECK: @f2_as1
while.body.i.preheader:
  %addr = getelementptr inbounds [10 x i8], [10 x i8] addrspace(1)* @global_as1, i16 0, i16 9
  br label %while.body.i

while.body.i:
; CHECK: phi
; CHECK-NEXT: phi
; CHECK-NOT: phi
  %c.addr.02.i = phi i8 addrspace(1)* [ %incdec.ptr.i, %while.body.i ], [ %addr, %while.body.i.preheader ]
  %incdec.ptr.i = getelementptr inbounds i8, i8 addrspace(1)* %c.addr.02.i, i16 -1
; CHECK: sub i16 10, %0
; CHECK-NEXT: icmp ult i16 10, %0
; CHECK-NEXT: icmp ult i16 {{.*}}, 1
; CHECK-NEXT: or i1
; CHECK-NEXT: br {{.*}}, label %trap
  store i8 100, i8 addrspace(1)* %c.addr.02.i, align 1
  %0 = load i8, i8 addrspace(1)* %incdec.ptr.i, align 1
  %tobool.i = icmp eq i8 %0, 0
  br i1 %tobool.i, label %fn.exit, label %while.body.i

fn.exit:
  ret void
}
