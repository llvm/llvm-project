; RUN: opt < %s -msan-check-access-address=0 -S -passes=msan 2>&1 | FileCheck  \
; RUN: %s
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=1 -S          \
; RUN: -passes=msan 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-ORIGIN
; RUN: opt < %s -msan-check-access-address=0 -S          \
; RUN: -passes="msan<track-origins=1>" 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-ORIGIN
; RUN: opt < %s -msan-check-access-address=0 -msan-track-origins=2 -S          \
; RUN: -passes=msan 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-ORIGIN

; Test that shadow and origin are stored for variadic function params.

target datalayout = "e-m:e-i32:32-f80:128-n8:16:32"
target triple = "i386-unknown-linux-gnu"

define dso_local i32 @test(i32 %a, i32 %b, i32 %c) local_unnamed_addr {
entry:
  %call = tail call i32 (i32, ...) @sum(i32 3, i32 %a, i32 %b, i32 %c)
  ret i32 %call
}

; CHECK: store i32 0, ptr @__msan_param_tls, align 8
; CHECK: store i32 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 8) to ptr), align 8
; CHECK: store i32 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 16) to ptr), align 8
; CHECK: store i32 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_param_tls to i64), i64 24) to ptr), align 8
; CHECK: store i32 0, ptr @__msan_va_arg_tls, align 8
; CHECK: store i32 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 8) to ptr), align 8
; CHECK: store i32 0, ptr inttoptr (i64 add (i64 ptrtoint (ptr @__msan_va_arg_tls to i64), i64 16) to ptr), align 8
; CHECK-ORIGIN: ore i64 24, ptr @__msan_va_arg_overflow_size_tls, align 4

define internal i32 @sum(i32 %n, ...) unnamed_addr #0 {
entry:
  %n.addr = alloca i32, align 4
  %args = alloca ptr, align 4
  %res = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  call void @llvm.va_start.p0(ptr %args)
  store i32 0, ptr %res, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %argp.cur = load ptr, ptr %args, align 4
  %argp.next = getelementptr inbounds i8, ptr %argp.cur, i32 4
  store ptr %argp.next, ptr %args, align 4
  %2 = load i32, ptr %argp.cur, align 4
  %3 = load i32, ptr %res, align 4
  %add = add nsw i32 %3, %2
  store i32 %add, ptr %res, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @llvm.va_end.p0(ptr %args)
  %5 = load i32, ptr %res, align 4
  ret i32 %5
}

; CHECK: call void @llvm.memcpy.{{.*}} [[SHADOW_COPY:%[_0-9a-z]+]], {{.*}} @__msan_va_arg_tls

; CHECK: call void @llvm.va_start.p0(ptr %args)
; CHECK: call void @llvm.memcpy.{{.*}}, {{.*}} [[SHADOW_COPY]], i{{.*}} [[REGSAVE:%[0-9]+]], i1 false)

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
