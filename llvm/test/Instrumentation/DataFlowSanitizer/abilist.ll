; RUN: opt < %s -passes=dfsan -dfsan-abilist=%S/Inputs/abilist.txt -S | FileCheck -enable-var-scope %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: i32 @discard(i32 %a, i32 %b)
define i32 @discard(i32 %a, i32 %b) {
  ret i32 0
}

; CHECK: i32 @functional(i32 %a, i32 %b)
define i32 @functional(i32 %a, i32 %b) {
  %c = add i32 %a, %b
  ret i32 %c
}

@discardg = alias ptr (i32), ptr @g

declare void @custom1(i32 %a, i32 %b)

declare i32 @custom2(i32 %a, i32 %b)

declare void @custom3(i32 %a, ...)

declare i32 @custom4(i32 %a, ...)

declare void @customcb(ptr %cb)

declare i32 @cb(i32)

; CHECK: @f.dfsan
define void @f(i32 %x) {
  ; CHECK: %[[LABELVA2:.*]] = alloca [2 x i8]
  ; CHECK: %[[LABELVA1:.*]] = alloca [2 x i8]
  ; CHECK: %[[LABELRETURN:.*]] = alloca i8

  ; CHECK: call void @__dfsw_custom1(i32 1, i32 2, i8 zeroext 0, i8 zeroext 0)
  call void @custom1(i32 1, i32 2)

  ; CHECK: call i32 @__dfsw_custom2(i32 1, i32 2, i8 zeroext 0, i8 zeroext 0, ptr %[[LABELRETURN]])
  call i32 @custom2(i32 1, i32 2)

  ; CHECK: call void @__dfsw_customcb({{.*}} @cb.dfsan, i8 zeroext 0)
  call void @customcb(ptr @cb)

  ; CHECK: %[[LABELVA1_0:.*]] = getelementptr inbounds nuw [2 x i8], ptr %[[LABELVA1]], i32 0, i32 0
  ; CHECK: store i8 0, ptr %[[LABELVA1_0]]
  ; CHECK: %[[LABELVA1_1:.*]] = getelementptr inbounds nuw [2 x i8], ptr %[[LABELVA1]], i32 0, i32 1
  ; CHECK: store i8 %{{.*}}, ptr %[[LABELVA1_1]]
  ; CHECK: %[[LABELVA1_0A:.*]] = getelementptr inbounds nuw [2 x i8], ptr %[[LABELVA1]], i32 0, i32 0
  ; CHECK: call void (i32, i8, ptr, ...) @__dfsw_custom3(i32 1, i8 zeroext 0, ptr %[[LABELVA1_0A]], i32 2, i32 %{{.*}})

  call void (i32, ...) @custom3(i32 1, i32 2, i32 %x)

  ; CHECK: %[[LABELVA2_0:.*]] = getelementptr inbounds nuw [2 x i8], ptr %[[LABELVA2]], i32 0, i32 0
  ; CHECK: %[[LABELVA2_0A:.*]] = getelementptr inbounds nuw [2 x i8], ptr %[[LABELVA2]], i32 0, i32 0
  ; CHECK: call i32 (i32, i8, ptr, ptr, ...) @__dfsw_custom4(i32 1, i8 zeroext 0, ptr %[[LABELVA2_0A]], ptr %[[LABELRETURN]], i32 2, i32 3)
  call i32 (i32, ...) @custom4(i32 1, i32 2, i32 3)

  ret void
}

; CHECK: @g.dfsan
define ptr @g(i32) {
  ; CHECK: ret {{.*}} @"dfsw$custom2"
  ret ptr @custom2
}

; CHECK: define ptr @discardg(i32 %0)
; CHECK: %[[CALL:.*]] = call ptr @g.dfsan(i32 %0)
; CHECK: load {{.*}} @__dfsan_retval_tls
; CHECK: ret {{.*}}

; CHECK: define i32 @adiscard.dfsan(i32 %0, i32 %1)
; CHECK: %[[CALL:.*]] = call i32 @discard(i32 %0, i32 %1)
; CHECK: ret i32
@adiscard = alias i32 (i32, i32), ptr @discard

; CHECK: define linkonce_odr i32 @"dfsw$custom2"(i32 %0, i32 %1)
; CHECK: %[[LABELRETURN2:.*]] = alloca i8
; CHECK: %[[RV:.*]] = call i32 @__dfsw_custom2(i32 {{.*}}, i32 {{.*}}, i8 {{.*}}, i8 {{.*}}, ptr %[[LABELRETURN2]])
; CHECK: %[[RVSHADOW:.*]] = load i8, ptr %[[LABELRETURN2]]
; CHECK: store {{.*}} @__dfsan_retval_tls
; CHECK: ret i32

; CHECK: define linkonce_odr void @"dfsw$custom3"(i32 %0, ...)
; CHECK: call void @__dfsan_vararg_wrapper(ptr
; CHECK: unreachable

; CHECK: define linkonce_odr i32 @"dfsw$custom4"(i32 %0, ...)

; CHECK: declare void @__dfsw_custom1(i32, i32, i8, i8)
; CHECK: declare i32 @__dfsw_custom2(i32, i32, i8, i8, ptr)

; CHECK: declare void @__dfsw_custom3(i32, i8, ptr, ...)
; CHECK: declare i32 @__dfsw_custom4(i32, i8, ptr, ptr, ...)
