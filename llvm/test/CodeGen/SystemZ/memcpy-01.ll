; Test memcpy using MVC.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.memcpy.p0i8.p0i8.i32(i8 *nocapture, i8 *nocapture, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8 *nocapture, i8 *nocapture, i64, i1) nounwind
declare void @foo(i8 *, i8 *)

; Test a no-op move, i32 version.
define void @f1(i8* %dest, i8* %src) {
; CHECK-LABEL: f1:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 0, i1 false)
  ret void
}

; Test a no-op move, i64 version.
define void @f2(i8* %dest, i8* %src) {
; CHECK-LABEL: f2:
; CHECK-NOT: %r2
; CHECK-NOT: %r3
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 0, i1 false)
  ret void
}

; Test a 1-byte move, i32 version.
define void @f3(i8* %dest, i8* %src) {
; CHECK-LABEL: f3:
; CHECK: mvc 0(1,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 1, i1 false)
  ret void
}

; Test a 1-byte move, i64 version.
define void @f4(i8* %dest, i8* %src) {
; CHECK-LABEL: f4:
; CHECK: mvc 0(1,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1, i1 false)
  ret void
}

; Test the upper range of a single MVC, i32 version.
define void @f5(i8* %dest, i8* %src) {
; CHECK-LABEL: f5:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 256, i1 false)
  ret void
}

; Test the upper range of a single MVC, i64 version.
define void @f6(i8* %dest, i8* %src) {
; CHECK-LABEL: f6:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 256, i1 false)
  ret void
}

; Test the first case that needs two MVCs.
define void @f7(i8* %dest, i8* %src) {
; CHECK-LABEL: f7:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: mvc 256(1,%r2), 256(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %dest, i8* %src, i32 257, i1 false)
  ret void
}

; Test the last-but-one case that needs two MVCs.
define void @f8(i8* %dest, i8* %src) {
; CHECK-LABEL: f8:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: mvc 256(255,%r2), 256(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 511, i1 false)
  ret void
}

; Test the last case that needs two MVCs.
define void @f9(i8* %dest, i8* %src) {
; CHECK-LABEL: f9:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: mvc 256(256,%r2), 256(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 512, i1 false)
  ret void
}

; Test an arbitrary value that uses straight-line code.
define void @f10(i8* %dest, i8* %src) {
; CHECK-LABEL: f10:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: mvc 256(256,%r2), 256(%r3)
; CHECK: mvc 512(256,%r2), 512(%r3)
; CHECK: mvc 768(256,%r2), 768(%r3)
; CHECK: mvc 1024(255,%r2), 1024(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1279, i1 false)
  ret void
}

; ...and again in cases where not all parts are in range of MVC.
define void @f11(i8* %srcbase, i8* %destbase) {
; CHECK-LABEL: f11:
; CHECK: mvc 4000(256,%r2), 3500(%r3)
; CHECK: lay [[NEWDEST:%r[1-5]]], 4256(%r2)
; CHECK: mvc 0(256,[[NEWDEST]]), 3756(%r3)
; CHECK: mvc 256(256,[[NEWDEST]]), 4012(%r3)
; CHECK: lay [[NEWSRC:%r[1-5]]], 4268(%r3)
; CHECK: mvc 512(256,[[NEWDEST]]), 0([[NEWSRC]])
; CHECK: mvc 768(255,[[NEWDEST]]), 256([[NEWSRC]])
; CHECK: br %r14
  %dest = getelementptr i8, i8* %srcbase, i64 4000
  %src = getelementptr i8, i8* %destbase, i64 3500
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1279, i1 false)
  ret void
}

; ...and again with a destination frame base that goes out of range.
define void @f12() {
; CHECK-LABEL: f12:
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc 4076(256,%r15), 2100(%r15)
; CHECK: lay [[NEWDEST:%r[1-5]]], 4332(%r15)
; CHECK: mvc 0(256,[[NEWDEST]]), 2356(%r15)
; CHECK: mvc 256(256,[[NEWDEST]]), 2612(%r15)
; CHECK: mvc 512(256,[[NEWDEST]]), 2868(%r15)
; CHECK: mvc 768(255,[[NEWDEST]]), 3124(%r15)
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %arr = alloca [6000 x i8]
  %dest = getelementptr [6000 x i8], [6000 x i8] *%arr, i64 0, i64 3900
  %src = getelementptr [6000 x i8], [6000 x i8] *%arr, i64 0, i64 1924
  call void @foo(i8* %dest, i8* %src)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1279, i1 false)
  call void @foo(i8* %dest, i8* %src)
  ret void
}

; ...and again with a source frame base that goes out of range.
define void @f13() {
; CHECK-LABEL: f13:
; CHECK: brasl %r14, foo@PLT
; CHECK: mvc 200(256,%r15), 3826(%r15)
; CHECK: mvc 456(256,%r15), 4082(%r15)
; CHECK: lay [[NEWSRC:%r[1-5]]], 4338(%r15)
; CHECK: mvc 712(256,%r15), 0([[NEWSRC]])
; CHECK: mvc 968(256,%r15), 256([[NEWSRC]])
; CHECK: mvc 1224(255,%r15), 512([[NEWSRC]])
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %arr = alloca [6000 x i8]
  %dest = getelementptr [6000 x i8], [6000 x i8] *%arr, i64 0, i64 24
  %src = getelementptr [6000 x i8], [6000 x i8] *%arr, i64 0, i64 3650
  call void @foo(i8* %dest, i8* %src)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1279, i1 false)
  call void @foo(i8* %dest, i8* %src)
  ret void
}

; Test the last case that is done using straight-line code.
define void @f14(i8* %dest, i8* %src) {
; CHECK-LABEL: f14:
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: mvc 256(256,%r2), 256(%r3)
; CHECK: mvc 512(256,%r2), 512(%r3)
; CHECK: mvc 768(256,%r2), 768(%r3)
; CHECK: mvc 1024(256,%r2), 1024(%r3)
; CHECK: mvc 1280(256,%r2), 1280(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1536, i1 false)
  ret void
}

; Test the first case that is done using a loop.
define void @f15(i8* %dest, i8* %src) {
; CHECK-LABEL: f15:
; CHECK: lghi [[COUNT:%r[0-5]]], 6
; CHECK: [[LABEL:\.L[^:]*]]:
; CHECK: pfd 2, 768(%r2)
; CHECK: mvc 0(256,%r2), 0(%r3)
; CHECK: la %r2, 256(%r2)
; CHECK: la %r3, 256(%r3)
; CHECK: brctg [[COUNT]], [[LABEL]]
; CHECK: mvc 0(1,%r2), 0(%r3)
; CHECK: br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1537, i1 false)
  ret void
}

; ...and again with frame bases, where the base must be loaded into a
; register before the loop.
define void @f16() {
; CHECK-LABEL: f16:
; CHECK: brasl %r14, foo@PLT
; CHECK-DAG: lghi [[COUNT:%r[0-5]]], 6
; CHECK-DAG: la [[BASE:%r[0-5]]], 160(%r15)
; CHECK: [[LABEL:\.L[^:]*]]:
; CHECK: pfd 2, 2368([[BASE]])
; CHECK: mvc 1600(256,[[BASE]]), 0([[BASE]])
; CHECK: la [[BASE]], 256([[BASE]])
; CHECK: brctg [[COUNT]], [[LABEL]]
; CHECK: mvc 1600(1,[[BASE]]), 0([[BASE]])
; CHECK: brasl %r14, foo@PLT
; CHECK: br %r14
  %arr = alloca [3200 x i8]
  %dest = getelementptr [3200 x i8], [3200 x i8] *%arr, i64 0, i64 1600
  %src = getelementptr [3200 x i8], [3200 x i8] *%arr, i64 0, i64 0
  call void @foo(i8* %dest, i8* %src)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 1537, i1 false)
  call void @foo(i8* %dest, i8* %src)
  ret void
}

; Test a variable length loop.
define void @f17(i8* %dest, i8* %src, i64 %Len) {
; CHECK-LABEL: f17:
; CHECK:       # %bb.0:
; CHECK-NEXT:    aghi %r4, -1
; CHECK-NEXT:    cgibe %r4, -1, 0(%r14)
; CHECK-NEXT:  .LBB16_1:
; CHECK-NEXT:    srlg %r0, %r4, 8
; CHECK-NEXT:    cgije %r0, 0, .LBB16_3
; CHECK-NEXT:  .LBB16_2: # =>This Inner Loop Header: Depth=1
; CHECK-NEXT:    pfd 2, 768(%r2)
; CHECK-NEXT:    mvc 0(256,%r2), 0(%r3)
; CHECK-NEXT:    la %r2, 256(%r2)
; CHECK-NEXT:    la %r3, 256(%r3)
; CHECK-NEXT:    brctg %r0, .LBB16_2
; CHECK-NEXT:  .LBB16_3:
; CHECK-NEXT:    exrl %r4, .Ltmp0
; CHECK-NEXT:    br %r14
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dest, i8* %src, i64 %Len, i1 false)
  ret void
}

; CHECK:       .Ltmp0:
; CHECK-NEXT:    mvc 0(1,%r2), 0(%r3)
