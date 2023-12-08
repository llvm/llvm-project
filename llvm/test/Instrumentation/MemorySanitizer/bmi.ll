; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; REQUIRES: x86-registered-target

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare i32 @llvm.x86.bmi.bzhi.32(i32, i32)
declare i32 @llvm.x86.bmi.bextr.32(i32, i32)
declare i32 @llvm.x86.bmi.pdep.32(i32, i32)
declare i32 @llvm.x86.bmi.pext.32(i32, i32)

declare i64 @llvm.x86.bmi.bzhi.64(i64, i64)
declare i64 @llvm.x86.bmi.bextr.64(i64, i64)
declare i64 @llvm.x86.bmi.pdep.64(i64, i64)
declare i64 @llvm.x86.bmi.pext.64(i64, i64)

define i32 @Test_bzhi_32(i32 %a, i32 %b) sanitize_memory {
entry:
  %c = tail call i32 @llvm.x86.bmi.bzhi.32(i32 %a, i32 %b)
  ret i32 %c
}

; CHECK-LABEL: @Test_bzhi_32(
; CHECK-DAG: %[[SA:.*]] = load i32, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i32, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i32 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i32
; CHECK-DAG: %[[X:.*]] = call i32 @llvm.x86.bmi.bzhi.32(i32 %[[SA]], i32 %b)
; CHECK-DAG: %[[S:.*]] = or i32 %[[SB1]], %[[X]]
; CHECK-DAG: store i32 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i32

define i64 @Test_bzhi_64(i64 %a, i64 %b) sanitize_memory {
entry:
  %c = tail call i64 @llvm.x86.bmi.bzhi.64(i64 %a, i64 %b)
  ret i64 %c
}

; CHECK-LABEL: @Test_bzhi_64(
; CHECK-DAG: %[[SA:.*]] = load i64, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i64, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i64 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i64
; CHECK-DAG: %[[X:.*]] = call i64 @llvm.x86.bmi.bzhi.64(i64 %[[SA]], i64 %b)
; CHECK-DAG: %[[S:.*]] = or i64 %[[SB1]], %[[X]]
; CHECK-DAG: store i64 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i64


define i32 @Test_bextr_32(i32 %a, i32 %b) sanitize_memory {
entry:
  %c = tail call i32 @llvm.x86.bmi.bextr.32(i32 %a, i32 %b)
  ret i32 %c
}

; CHECK-LABEL: @Test_bextr_32(
; CHECK-DAG: %[[SA:.*]] = load i32, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i32, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i32 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i32
; CHECK-DAG: %[[X:.*]] = call i32 @llvm.x86.bmi.bextr.32(i32 %[[SA]], i32 %b)
; CHECK-DAG: %[[S:.*]] = or i32 %[[SB1]], %[[X]]
; CHECK-DAG: store i32 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i32

define i64 @Test_bextr_64(i64 %a, i64 %b) sanitize_memory {
entry:
  %c = tail call i64 @llvm.x86.bmi.bextr.64(i64 %a, i64 %b)
  ret i64 %c
}

; CHECK-LABEL: @Test_bextr_64(
; CHECK-DAG: %[[SA:.*]] = load i64, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i64, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i64 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i64
; CHECK-DAG: %[[X:.*]] = call i64 @llvm.x86.bmi.bextr.64(i64 %[[SA]], i64 %b)
; CHECK-DAG: %[[S:.*]] = or i64 %[[SB1]], %[[X]]
; CHECK-DAG: store i64 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i64


define i32 @Test_pdep_32(i32 %a, i32 %b) sanitize_memory {
entry:
  %c = tail call i32 @llvm.x86.bmi.pdep.32(i32 %a, i32 %b)
  ret i32 %c
}

; CHECK-LABEL: @Test_pdep_32(
; CHECK-DAG: %[[SA:.*]] = load i32, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i32, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i32 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i32
; CHECK-DAG: %[[X:.*]] = call i32 @llvm.x86.bmi.pdep.32(i32 %[[SA]], i32 %b)
; CHECK-DAG: %[[S:.*]] = or i32 %[[SB1]], %[[X]]
; CHECK-DAG: store i32 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i32

define i64 @Test_pdep_64(i64 %a, i64 %b) sanitize_memory {
entry:
  %c = tail call i64 @llvm.x86.bmi.pdep.64(i64 %a, i64 %b)
  ret i64 %c
}

; CHECK-LABEL: @Test_pdep_64(
; CHECK-DAG: %[[SA:.*]] = load i64, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i64, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i64 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i64
; CHECK-DAG: %[[X:.*]] = call i64 @llvm.x86.bmi.pdep.64(i64 %[[SA]], i64 %b)
; CHECK-DAG: %[[S:.*]] = or i64 %[[SB1]], %[[X]]
; CHECK-DAG: store i64 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i64

define i32 @Test_pext_32(i32 %a, i32 %b) sanitize_memory {
entry:
  %c = tail call i32 @llvm.x86.bmi.pext.32(i32 %a, i32 %b)
  ret i32 %c
}

; CHECK-LABEL: @Test_pext_32(
; CHECK-DAG: %[[SA:.*]] = load i32, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i32, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i32 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i32
; CHECK-DAG: %[[X:.*]] = call i32 @llvm.x86.bmi.pext.32(i32 %[[SA]], i32 %b)
; CHECK-DAG: %[[S:.*]] = or i32 %[[SB1]], %[[X]]
; CHECK-DAG: store i32 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i32

define i64 @Test_pext_64(i64 %a, i64 %b) sanitize_memory {
entry:
  %c = tail call i64 @llvm.x86.bmi.pext.64(i64 %a, i64 %b)
  ret i64 %c
}

; CHECK-LABEL: @Test_pext_64(
; CHECK-DAG: %[[SA:.*]] = load i64, ptr @__msan_param_tls
; CHECK-DAG: %[[SB:.*]] = load i64, {{.*}}@__msan_param_tls to i64), i64 8)
; CHECK-DAG: %[[SB0:.*]] = icmp ne i64 %[[SB]], 0
; CHECK-DAG: %[[SB1:.*]] = sext i1 %[[SB0]] to i64
; CHECK-DAG: %[[X:.*]] = call i64 @llvm.x86.bmi.pext.64(i64 %[[SA]], i64 %b)
; CHECK-DAG: %[[S:.*]] = or i64 %[[SB1]], %[[X]]
; CHECK-DAG: store i64 %[[S]], {{.*}}@__msan_retval_tls
; CHECK: ret i64
