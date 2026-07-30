; RUN: opt < %s -passes=asan -asan-use-after-scope -asan-use-stack-safety=1 -S | FileCheck %s

; Source (-O0 -fsanitize=address -fsanitize-address-use-after-scope):
;; struct S { int x, y; };
;; void swap(S *a, S *b, bool doit) {
;;   if (!doit)
;;     return;
;;   auto tmp = *a;
;;   *a = *b;
;;   *b = tmp;
;; }

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.14.0"

%struct.S = type { i32, i32 }

; CHECK-LABEL: define {{.*}} @_Z4swapP1SS0_b(

; First come the argument allocas.
; CHECK:      %a.addr = alloca ptr, align 8
; CHECK-NEXT: %b.addr = alloca ptr, align 8
; CHECK-NEXT: %doit.addr = alloca i8, align 1

; Next, the stores into the argument allocas.
; CHECK-NEXT: store ptr {{.*}}, ptr %a.addr
; CHECK-NEXT: store ptr {{.*}}, ptr %b.addr
; CHECK-NEXT: [[frombool:%.*]] = zext i1 {{.*}} to i8
; CHECK-NEXT: store i8 %frombool, ptr %doit.addr, align 1
; CHECK-NEXT: [[stack_base:%.*]] = alloca i64, align 8

define void @_Z4swapP1SS0_b(ptr %a, ptr %b, i1 zeroext %doit) sanitize_address {
entry:
  %a.addr = alloca ptr, align 8
  %b.addr = alloca ptr, align 8
  %doit.addr = alloca i8, align 1
  %tmp = alloca %struct.S, align 4
  store ptr %a, ptr %a.addr, align 8
  store ptr %b, ptr %b.addr, align 8
  %frombool = zext i1 %doit to i8
  store i8 %frombool, ptr %doit.addr, align 1
  %0 = load i8, ptr %doit.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %1 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp, ptr align 4 %1, i64 8, i1 false)
  %2 = load ptr, ptr %b.addr, align 8
  %3 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %3, ptr align 4 %2, i64 8, i1 false)
  %4 = load ptr, ptr %b.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %4, ptr align 4 %tmp, i64 8, i1 false)
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret void
}

; Synthetic test case, meant to check that we do not reorder instructions past
; a load when attempting to hoist argument init insts.
; CHECK-LABEL: define {{.*}} @func_with_load_in_arginit_sequence
; CHECK:      [[argA:%.*]] = alloca ptr,
; CHECK-NEXT: [[argB:%.*]] = alloca ptr,
; CHECK-NEXT: [[argDoit:%.*]] = alloca i8,
; CHECK-NEXT: %tmp = alloca %struct.S, align 4
; CHECK-NEXT: store ptr {{.*}}, ptr [[argA]]
; CHECK-NEXT: store ptr {{.*}}, ptr [[argB]]
; CHECK-NEXT: %0 = load i8, ptr %doit.addr, align 1
; CHECK-NEXT: %frombool = zext i1 %doit to i8
; CHECK-NEXT: store i8 %frombool, ptr %doit.addr, align 1
define void @func_with_load_in_arginit_sequence(ptr %a, ptr %b, i1 zeroext %doit) sanitize_address {
entry:
  %a.addr = alloca ptr, align 8
  %b.addr = alloca ptr, align 8
  %doit.addr = alloca i8, align 1
  %tmp = alloca %struct.S, align 4
  store ptr %a, ptr %a.addr, align 8
  store ptr %b, ptr %b.addr, align 8

  ; This load prevents the next argument init sequence from being moved.
  %0 = load i8, ptr %doit.addr, align 1 

  %frombool = zext i1 %doit to i8
  store i8 %frombool, ptr %doit.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %1 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp, ptr align 4 %1, i64 8, i1 false)
  %2 = load ptr, ptr %b.addr, align 8
  %3 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %3, ptr align 4 %2, i64 8, i1 false)
  %4 = load ptr, ptr %b.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %4, ptr align 4 %tmp, i64 8, i1 false)
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret void
}

; Synthetic test case, meant to check that we can handle functions with more
; than one interesting alloca.
; CHECK-LABEL: define {{.*}} @func_with_multiple_interesting_allocas
; CHECK:      [[argA:%.*]] = alloca ptr,
; CHECK-NEXT: [[argB:%.*]] = alloca ptr,
; CHECK-NEXT: [[argDoit:%.*]] = alloca i8,
; CHECK-NEXT: %tmp = alloca %struct.S, align 4
; CHECK-NEXT: %tmp2 = alloca %struct.S, align 4
; CHECK-NEXT: store ptr {{.*}}, ptr [[argA]]
; CHECK-NEXT: store ptr {{.*}}, ptr [[argB]]
; CHECK-NEXT: [[frombool:%.*]] = zext i1 {{.*}} to i8
; CHECK-NEXT: store i8 [[frombool]], ptr [[argDoit]]
; CHECK-NEXT: %0 = load i8, ptr %doit.addr, align 1
define void @func_with_multiple_interesting_allocas(ptr %a, ptr %b, i1 zeroext %doit) sanitize_address {
entry:
  %a.addr = alloca ptr, align 8
  %b.addr = alloca ptr, align 8
  %doit.addr = alloca i8, align 1
  %tmp = alloca %struct.S, align 4
  %tmp2 = alloca %struct.S, align 4
  store ptr %a, ptr %a.addr, align 8
  store ptr %b, ptr %b.addr, align 8
  %frombool = zext i1 %doit to i8
  store i8 %frombool, ptr %doit.addr, align 1
  %0 = load i8, ptr %doit.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %1 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp, ptr align 4 %1, i64 8, i1 false)
  %2 = load ptr, ptr %b.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %tmp2, ptr align 4 %2, i64 8, i1 false)
  %3 = load ptr, ptr %b.addr, align 8
  %4 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %4, ptr align 4 %3, i64 8, i1 false)
  %5 = load ptr, ptr %b.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %5, ptr align 4 %tmp, i64 8, i1 false)
  %6 = load ptr, ptr %a.addr, align 8
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %6, ptr align 4 %tmp2, i64 8, i1 false)
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret void
}

declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg)
