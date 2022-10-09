; RUN: opt < %s -passes=asan -S | FileCheck %s

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
; CHECK:      [[argA:%.*]] = alloca %struct.S*,
; CHECK-NEXT: [[argB:%.*]] = alloca %struct.S*,
; CHECK-NEXT: [[argDoit:%.*]] = alloca i8,

; Next, the stores into the argument allocas.
; CHECK-NEXT: store %struct.S* {{.*}}, %struct.S** [[argA]]
; CHECK-NEXT: store %struct.S* {{.*}}, %struct.S** [[argB]]
; CHECK-NEXT: [[frombool:%.*]] = zext i1 {{.*}} to i8
; CHECK-NEXT: store i8 [[frombool]], i8* [[argDoit]]

define void @_Z4swapP1SS0_b(%struct.S* %a, %struct.S* %b, i1 zeroext %doit) sanitize_address {
entry:
  %a.addr = alloca %struct.S*, align 8
  %b.addr = alloca %struct.S*, align 8
  %doit.addr = alloca i8, align 1
  %tmp = alloca %struct.S, align 4
  store %struct.S* %a, %struct.S** %a.addr, align 8
  store %struct.S* %b, %struct.S** %b.addr, align 8
  %frombool = zext i1 %doit to i8
  store i8 %frombool, i8* %doit.addr, align 1
  %0 = load i8, i8* %doit.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %1 = load %struct.S*, %struct.S** %a.addr, align 8
  %2 = bitcast %struct.S* %tmp to i8*
  %3 = bitcast %struct.S* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 8, i1 false)
  %4 = load %struct.S*, %struct.S** %b.addr, align 8
  %5 = load %struct.S*, %struct.S** %a.addr, align 8
  %6 = bitcast %struct.S* %5 to i8*
  %7 = bitcast %struct.S* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 %7, i64 8, i1 false)
  %8 = load %struct.S*, %struct.S** %b.addr, align 8
  %9 = bitcast %struct.S* %8 to i8*
  %10 = bitcast %struct.S* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 8, i1 false)
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret void
}

; Synthetic test case, meant to check that we do not reorder instructions past
; a load when attempting to hoist argument init insts.
; CHECK-LABEL: define {{.*}} @func_with_load_in_arginit_sequence
; CHECK:      [[argA:%.*]] = alloca %struct.S*,
; CHECK-NEXT: [[argB:%.*]] = alloca %struct.S*,
; CHECK-NEXT: [[argDoit:%.*]] = alloca i8,
; CHECK-NEXT: store %struct.S* {{.*}}, %struct.S** [[argA]]
; CHECK-NEXT: store %struct.S* {{.*}}, %struct.S** [[argB]]
; CHECK-NEXT: [[stack_base:%.*]] = alloca i64
define void @func_with_load_in_arginit_sequence(%struct.S* %a, %struct.S* %b, i1 zeroext %doit) sanitize_address {
entry:
  %a.addr = alloca %struct.S*, align 8
  %b.addr = alloca %struct.S*, align 8
  %doit.addr = alloca i8, align 1
  %tmp = alloca %struct.S, align 4
  store %struct.S* %a, %struct.S** %a.addr, align 8
  store %struct.S* %b, %struct.S** %b.addr, align 8

  ; This load prevents the next argument init sequence from being moved.
  %0 = load i8, i8* %doit.addr, align 1 

  %frombool = zext i1 %doit to i8
  store i8 %frombool, i8* %doit.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %1 = load %struct.S*, %struct.S** %a.addr, align 8
  %2 = bitcast %struct.S* %tmp to i8*
  %3 = bitcast %struct.S* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 8, i1 false)
  %4 = load %struct.S*, %struct.S** %b.addr, align 8
  %5 = load %struct.S*, %struct.S** %a.addr, align 8
  %6 = bitcast %struct.S* %5 to i8*
  %7 = bitcast %struct.S* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %6, i8* align 4 %7, i64 8, i1 false)
  %8 = load %struct.S*, %struct.S** %b.addr, align 8
  %9 = bitcast %struct.S* %8 to i8*
  %10 = bitcast %struct.S* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 8, i1 false)
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret void
}

; Synthetic test case, meant to check that we can handle functions with more
; than one interesting alloca.
; CHECK-LABEL: define {{.*}} @func_with_multiple_interesting_allocas
; CHECK:      [[argA:%.*]] = alloca %struct.S*,
; CHECK-NEXT: [[argB:%.*]] = alloca %struct.S*,
; CHECK-NEXT: [[argDoit:%.*]] = alloca i8,
; CHECK-NEXT: store %struct.S* {{.*}}, %struct.S** [[argA]]
; CHECK-NEXT: store %struct.S* {{.*}}, %struct.S** [[argB]]
; CHECK-NEXT: [[frombool:%.*]] = zext i1 {{.*}} to i8
; CHECK-NEXT: store i8 [[frombool]], i8* [[argDoit]]
define void @func_with_multiple_interesting_allocas(%struct.S* %a, %struct.S* %b, i1 zeroext %doit) sanitize_address {
entry:
  %a.addr = alloca %struct.S*, align 8
  %b.addr = alloca %struct.S*, align 8
  %doit.addr = alloca i8, align 1
  %tmp = alloca %struct.S, align 4
  %tmp2 = alloca %struct.S, align 4
  store %struct.S* %a, %struct.S** %a.addr, align 8
  store %struct.S* %b, %struct.S** %b.addr, align 8
  %frombool = zext i1 %doit to i8
  store i8 %frombool, i8* %doit.addr, align 1
  %0 = load i8, i8* %doit.addr, align 1
  %tobool = trunc i8 %0 to i1
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %entry
  br label %return

if.end:                                           ; preds = %entry
  %1 = load %struct.S*, %struct.S** %a.addr, align 8
  %2 = bitcast %struct.S* %tmp to i8*
  %3 = bitcast %struct.S* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %2, i8* align 4 %3, i64 8, i1 false)
  %4 = load %struct.S*, %struct.S** %b.addr, align 8
  %5 = bitcast %struct.S* %tmp2 to i8*
  %6 = bitcast %struct.S* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %5, i8* align 4 %6, i64 8, i1 false)
  %7 = load %struct.S*, %struct.S** %b.addr, align 8
  %8 = load %struct.S*, %struct.S** %a.addr, align 8
  %9 = bitcast %struct.S* %8 to i8*
  %10 = bitcast %struct.S* %7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %9, i8* align 4 %10, i64 8, i1 false)
  %11 = load %struct.S*, %struct.S** %b.addr, align 8
  %12 = bitcast %struct.S* %11 to i8*
  %13 = bitcast %struct.S* %tmp to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %12, i8* align 4 %13, i64 8, i1 false)
  %14 = load %struct.S*, %struct.S** %a.addr, align 8
  %15 = bitcast %struct.S* %14 to i8*
  %16 = bitcast %struct.S* %tmp2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %15, i8* align 4 %16, i64 8, i1 false)
  br label %return

return:                                           ; preds = %if.end, %if.then
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg)
