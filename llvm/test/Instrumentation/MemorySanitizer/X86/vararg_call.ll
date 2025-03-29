; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,CHECK-ORIGIN"
; RUN: opt < %s -msan-check-access-address=0 -S -passes="msan<track-origins=1>" 2>&1 | FileCheck %s "--check-prefixes=CHECK,CHECK-ORIGIN"
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S -passes=msan 2>&1 | FileCheck %s "--check-prefixes=CHECK,CHECK-ORIGIN"

; Test that shadow and origin are stored for variadic function params.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.__va_list_tag = type { i32, i32, ptr, ptr }

define dso_local i32 @test(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
  %call = tail call i32 (i32, ...) @sum(i32 3, i32 %a, i32 %b, i32 %c)
  ret i32 %call
}

; CHECK: store i32 0, {{.*}} @__msan_param_tls {{.*}} i64 8
; CHECK: store i32 0, {{.*}} @__msan_param_tls {{.*}} i64 16
; CHECK: store i32 0, {{.*}} @__msan_param_tls {{.*}} i64 24
; CHECK: store i32 0, {{.*}} @__msan_va_arg_tls {{.*}} i64 8
; CHECK-ORIGIN: store i32 0, {{.*}} @__msan_va_arg_origin_tls {{.*}} i64 8
; CHECK: store i32 0, {{.*}} @__msan_va_arg_tls {{.*}} i64 16
; CHECK-ORIGIN: store i32 0, {{.*}} @__msan_va_arg_origin_tls {{.*}} i64 16
; CHECK: store i32 0, {{.*}} @__msan_va_arg_tls {{.*}} i64 24
; CHECK-ORIGIN: store i32 0, {{.*}} @__msan_va_arg_origin_tls {{.*}} i64 24

define dso_local i32 @sum(i32 %n, ...) local_unnamed_addr #0 {
entry:
  %args = alloca [1 x %struct.__va_list_tag], align 16
  call void @llvm.lifetime.start.p0(i64 24, ptr nonnull %args) #2
  call void @llvm.va_start(ptr nonnull %args)
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.lr.ph, label %for.end

; CHECK: call void @llvm.memcpy.{{.*}} [[SHADOW_COPY:%[_0-9a-z]+]], {{.*}} @__msan_va_arg_tls
; CHECK-ORIGIN: call void @llvm.memcpy{{.*}} [[ORIGIN_COPY:%[_0-9a-z]+]], {{.*}} @__msan_va_arg_origin_tls

; CHECK: call void @llvm.va_start
; CHECK: call void @llvm.memcpy.{{.*}}, {{.*}} [[SHADOW_COPY]], i{{.*}} [[REGSAVE:[0-9]+]]
; CHECK-ORIGIN: call void @llvm.memcpy.{{.*}}, {{.*}} [[ORIGIN_COPY]], i{{.*}} [[REGSAVE]]

; CHECK: [[OVERFLOW_SHADOW:%[_0-9a-z]+]] = getelementptr i8, ptr [[SHADOW_COPY]], i{{.*}} [[REGSAVE]]
; CHECK: call void @llvm.memcpy.{{.*}}[[OVERFLOW_SHADOW]]
; CHECK-ORIGIN: [[OVERFLOW_ORIGIN:%[_0-9a-z]+]] = getelementptr i8, ptr [[ORIGIN_COPY]], i{{.*}} [[REGSAVE]]
; CHECK-ORIGIN: call void @llvm.memcpy.{{.*}}[[OVERFLOW_ORIGIN]]

for.body.lr.ph:                                   ; preds = %entry
  %0 = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %args, i64 0, i64 0, i32 3
  %overflow_arg_area_p = getelementptr inbounds [1 x %struct.__va_list_tag], ptr %args, i64 0, i64 0, i32 2
  %gp_offset.pre = load i32, ptr %args, align 16
  br label %for.body

for.body:                                         ; preds = %vaarg.end, %for.body.lr.ph
  %gp_offset = phi i32 [ %gp_offset.pre, %for.body.lr.ph ], [ %gp_offset12, %vaarg.end ]
  %sum.011 = phi i32 [ 0, %for.body.lr.ph ], [ %add, %vaarg.end ]
  %i.010 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %vaarg.end ]
  %fits_in_gp = icmp ult i32 %gp_offset, 41
  br i1 %fits_in_gp, label %vaarg.in_reg, label %vaarg.in_mem

vaarg.in_reg:                                     ; preds = %for.body
  %reg_save_area = load ptr, ptr %0, align 16
  %1 = sext i32 %gp_offset to i64
  %2 = getelementptr i8, ptr %reg_save_area, i64 %1
  %3 = add i32 %gp_offset, 8
  store i32 %3, ptr %args, align 16
  br label %vaarg.end

vaarg.in_mem:                                     ; preds = %for.body
  %overflow_arg_area = load ptr, ptr %overflow_arg_area_p, align 8
  %overflow_arg_area.next = getelementptr i8, ptr %overflow_arg_area, i64 8
  store ptr %overflow_arg_area.next, ptr %overflow_arg_area_p, align 8
  br label %vaarg.end

vaarg.end:                                        ; preds = %vaarg.in_mem, %vaarg.in_reg
  %gp_offset12 = phi i32 [ %3, %vaarg.in_reg ], [ %gp_offset, %vaarg.in_mem ]
  %vaarg.addr.in = phi ptr [ %2, %vaarg.in_reg ], [ %overflow_arg_area, %vaarg.in_mem ]
  %4 = load i32, ptr %vaarg.addr.in, align 4
  %add = add nsw i32 %4, %sum.011
  %inc = add nuw nsw i32 %i.010, 1
  %exitcond = icmp eq i32 %inc, %n
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %vaarg.end, %entry
  %sum.0.lcssa = phi i32 [ 0, %entry ], [ %add, %vaarg.end ]
  call void @llvm.va_end(ptr nonnull %args)
  call void @llvm.lifetime.end.p0(i64 24, ptr nonnull %args) #2
  ret i32 %sum.0.lcssa
}


; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.start.p0(i64, ptr nocapture) #1

; Function Attrs: nounwind
declare void @llvm.va_start(ptr) #2

; Function Attrs: nounwind
declare void @llvm.va_end(ptr) #2

; Function Attrs: argmemonly nounwind
declare void @llvm.lifetime.end.p0(i64, ptr nocapture) #1

declare dso_local i80 @sum_i80(i32, ...) local_unnamed_addr

; Unaligned types like i80 should also work.
define dso_local i80 @test_i80(i80 %a, i80 %b, i80 %c) local_unnamed_addr {
entry:
  %call = tail call i80 (i32, ...) @sum_i80(i32 3, i80 %a, i80 %b, i80 %c)
  ret i80 %call
}

