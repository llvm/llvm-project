; RUN: llc < %s -march=nvptx -mcpu=sm_20 | FileCheck %s
; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx -mcpu=sm_20 | %ptxas-verify %}
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

;; These tests should run for all targets

;;===-- Basic instruction selection tests ---------------------------------===;;


;;; i64

define i64 @icmp_eq_i64(i64 %a, i64 %b) {
; CHECK: setp.eq.s64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp eq i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_ne_i64(i64 %a, i64 %b) {
; CHECK: setp.ne.s64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ne i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_ugt_i64(i64 %a, i64 %b) {
; CHECK: setp.gt.u64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ugt i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_uge_i64(i64 %a, i64 %b) {
; CHECK: setp.ge.u64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp uge i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_ult_i64(i64 %a, i64 %b) {
; CHECK: setp.lt.u64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ult i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_ule_i64(i64 %a, i64 %b) {
; CHECK: setp.le.u64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ule i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_sgt_i64(i64 %a, i64 %b) {
; CHECK: setp.gt.s64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sgt i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_sge_i64(i64 %a, i64 %b) {
; CHECK: setp.ge.s64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sge i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_slt_i64(i64 %a, i64 %b) {
; CHECK: setp.lt.s64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp slt i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

define i64 @icmp_sle_i64(i64 %a, i64 %b) {
; CHECK: setp.le.s64 %p[[P0:[0-9]+]], %rd{{[0-9]+}}, %rd{{[0-9]+}}
; CHECK: selp.u64 %rd{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sle i64 %a, %b
  %ret = zext i1 %cmp to i64
  ret i64 %ret
}

;;; i32

define i32 @icmp_eq_i32(i32 %a, i32 %b) {
; CHECK: setp.eq.s32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp eq i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_ne_i32(i32 %a, i32 %b) {
; CHECK: setp.ne.s32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ne i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_ugt_i32(i32 %a, i32 %b) {
; CHECK: setp.gt.u32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ugt i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_uge_i32(i32 %a, i32 %b) {
; CHECK: setp.ge.u32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp uge i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_ult_i32(i32 %a, i32 %b) {
; CHECK: setp.lt.u32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ult i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_ule_i32(i32 %a, i32 %b) {
; CHECK: setp.le.u32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ule i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_sgt_i32(i32 %a, i32 %b) {
; CHECK: setp.gt.s32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sgt i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_sge_i32(i32 %a, i32 %b) {
; CHECK: setp.ge.s32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sge i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_slt_i32(i32 %a, i32 %b) {
; CHECK: setp.lt.s32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp slt i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}

define i32 @icmp_sle_i32(i32 %a, i32 %b) {
; CHECK: setp.le.s32 %p[[P0:[0-9]+]], %r{{[0-9]+}}, %r{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sle i32 %a, %b
  %ret = zext i1 %cmp to i32
  ret i32 %ret
}


;;; i16

define i16 @icmp_eq_i16(i16 %a, i16 %b) {
; CHECK: setp.eq.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp eq i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_ne_i16(i16 %a, i16 %b) {
; CHECK: setp.ne.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ne i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_ugt_i16(i16 %a, i16 %b) {
; CHECK: setp.gt.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ugt i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_uge_i16(i16 %a, i16 %b) {
; CHECK: setp.ge.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp uge i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_ult_i16(i16 %a, i16 %b) {
; CHECK: setp.lt.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ult i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_ule_i16(i16 %a, i16 %b) {
; CHECK: setp.le.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ule i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_sgt_i16(i16 %a, i16 %b) {
; CHECK: setp.gt.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sgt i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_sge_i16(i16 %a, i16 %b) {
; CHECK: setp.ge.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sge i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_slt_i16(i16 %a, i16 %b) {
; CHECK: setp.lt.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp slt i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}

define i16 @icmp_sle_i16(i16 %a, i16 %b) {
; CHECK: setp.le.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sle i16 %a, %b
  %ret = zext i1 %cmp to i16
  ret i16 %ret
}


;;; i8

define i8 @icmp_eq_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.eq.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp eq i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_ne_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.ne.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ne i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_ugt_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.gt.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ugt i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_uge_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.ge.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp uge i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_ult_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.lt.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ult i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_ule_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.le.u16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp ule i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_sgt_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.gt.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sgt i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_sge_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.ge.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sge i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_slt_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.lt.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp slt i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}

define i8 @icmp_sle_i8(i8 %a, i8 %b) {
; Comparison happens in 16-bit
; CHECK: setp.le.s16 %p[[P0:[0-9]+]], %rs{{[0-9]+}}, %rs{{[0-9]+}}
; CHECK: selp.u32 %r{{[0-9]+}}, 1, 0, %p[[P0]]
; CHECK: ret
  %cmp = icmp sle i8 %a, %b
  %ret = zext i1 %cmp to i8
  ret i8 %ret
}
