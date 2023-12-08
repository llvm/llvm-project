; Test to make sure NVVM intrinsics are automatically upgraded.
; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

declare i32 @llvm.nvvm.brev32(i32)
declare i64 @llvm.nvvm.brev64(i64)
declare i32 @llvm.nvvm.clz.i(i32)
declare i32 @llvm.nvvm.clz.ll(i64)
declare i32 @llvm.nvvm.popc.i(i32)
declare i32 @llvm.nvvm.popc.ll(i64)
declare float @llvm.nvvm.h2f(i16)

declare i32 @llvm.nvvm.abs.i(i32)
declare i64 @llvm.nvvm.abs.ll(i64)

declare i32 @llvm.nvvm.max.i(i32, i32)
declare i64 @llvm.nvvm.max.ll(i64, i64)
declare i32 @llvm.nvvm.max.ui(i32, i32)
declare i64 @llvm.nvvm.max.ull(i64, i64)
declare i32 @llvm.nvvm.min.i(i32, i32)
declare i64 @llvm.nvvm.min.ll(i64, i64)
declare i32 @llvm.nvvm.min.ui(i32, i32)
declare i64 @llvm.nvvm.min.ull(i64, i64)

; CHECK-LABEL: @simple_upgrade
define void @simple_upgrade(i32 %a, i64 %b, i16 %c) {
; CHECK: call i32 @llvm.bitreverse.i32(i32 %a)
  %r1 = call i32 @llvm.nvvm.brev32(i32 %a)

; CHECK: call i64 @llvm.bitreverse.i64(i64 %b)
  %r2 = call i64 @llvm.nvvm.brev64(i64 %b)

; CHECK: call i32 @llvm.ctlz.i32(i32 %a, i1 false)
  %r3 = call i32 @llvm.nvvm.clz.i(i32 %a)

; CHECK: [[clz:%[a-zA-Z0-9.]+]] = call i64 @llvm.ctlz.i64(i64 %b, i1 false)
; CHECK: trunc i64 [[clz]] to i32
  %r4 = call i32 @llvm.nvvm.clz.ll(i64 %b)

; CHECK: call i32 @llvm.ctpop.i32(i32 %a)
  %r5 = call i32 @llvm.nvvm.popc.i(i32 %a)

; CHECK: [[popc:%[a-zA-Z0-9.]+]] = call i64 @llvm.ctpop.i64(i64 %b)
; CHECK: trunc i64 [[popc]] to i32
  %r6 = call i32 @llvm.nvvm.popc.ll(i64 %b)

; CHECK: call float @llvm.convert.from.fp16.f32(i16 %c)
  %r7 = call float @llvm.nvvm.h2f(i16 %c)
  ret void
}

; CHECK-LABEL: @abs
define void @abs(i32 %a, i64 %b) {
; CHECK-DAG: [[negi:%[a-zA-Z0-9.]+]] = sub i32 0, %a
; CHECK-DAG: [[cmpi:%[a-zA-Z0-9.]+]] = icmp sge i32 %a, 0
; CHECK: select i1 [[cmpi]], i32 %a, i32 [[negi]]
  %r1 = call i32 @llvm.nvvm.abs.i(i32 %a)

; CHECK-DAG: [[negll:%[a-zA-Z0-9.]+]] = sub i64 0, %b
; CHECK-DAG: [[cmpll:%[a-zA-Z0-9.]+]] = icmp sge i64 %b, 0
; CHECK: select i1 [[cmpll]], i64 %b, i64 [[negll]]
  %r2 = call i64 @llvm.nvvm.abs.ll(i64 %b)

  ret void
}

; CHECK-LABEL: @min_max
define void @min_max(i32 %a1, i32 %a2, i64 %b1, i64 %b2) {
; CHECK: [[maxi:%[a-zA-Z0-9.]+]] = icmp sge i32 %a1, %a2
; CHECK: select i1 [[maxi]], i32 %a1, i32 %a2
  %r1 = call i32 @llvm.nvvm.max.i(i32 %a1, i32 %a2)

; CHECK: [[maxll:%[a-zA-Z0-9.]+]] = icmp sge i64 %b1, %b2
; CHECK: select i1 [[maxll]], i64 %b1, i64 %b2
  %r2 = call i64 @llvm.nvvm.max.ll(i64 %b1, i64 %b2)

; CHECK: [[maxui:%[a-zA-Z0-9.]+]] = icmp uge i32 %a1, %a2
; CHECK: select i1 [[maxui]], i32 %a1, i32 %a2
  %r3 = call i32 @llvm.nvvm.max.ui(i32 %a1, i32 %a2)

; CHECK: [[maxull:%[a-zA-Z0-9.]+]] = icmp uge i64 %b1, %b2
; CHECK: select i1 [[maxull]], i64 %b1, i64 %b2
  %r4 = call i64 @llvm.nvvm.max.ull(i64 %b1, i64 %b2)

; CHECK: [[mini:%[a-zA-Z0-9.]+]] = icmp sle i32 %a1, %a2
; CHECK: select i1 [[mini]], i32 %a1, i32 %a2
  %r5 = call i32 @llvm.nvvm.min.i(i32 %a1, i32 %a2)

; CHECK: [[minll:%[a-zA-Z0-9.]+]] = icmp sle i64 %b1, %b2
; CHECK: select i1 [[minll]], i64 %b1, i64 %b2
  %r6 = call i64 @llvm.nvvm.min.ll(i64 %b1, i64 %b2)

; CHECK: [[minui:%[a-zA-Z0-9.]+]] = icmp ule i32 %a1, %a2
; CHECK: select i1 [[minui]], i32 %a1, i32 %a2
  %r7 = call i32 @llvm.nvvm.min.ui(i32 %a1, i32 %a2)

; CHECK: [[minull:%[a-zA-Z0-9.]+]] = icmp ule i64 %b1, %b2
; CHECK: select i1 [[minull]], i64 %b1, i64 %b2
  %r8 = call i64 @llvm.nvvm.min.ull(i64 %b1, i64 %b2)

  ret void
}
