; RUN: opt < %s -memcpyopt -S | FileCheck %s
; Make sure memcpy-memcpy dependence is optimized across
; basic blocks (conditional branches and invokes).

%struct.s = type { i32, i32 }

@s_foo = private unnamed_addr constant %struct.s { i32 1, i32 2 }, align 4
@s_baz = private unnamed_addr constant %struct.s { i32 1, i32 2 }, align 4
@i = external constant i8*

declare void @qux()
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i32, i1)
declare void @__cxa_throw(i8*, i8*, i8*)
declare i32 @__gxx_personality_v0(...)
declare i8* @__cxa_begin_catch(i8*)

; A simple partial redundancy. Test that the second memcpy is optimized
; to copy directly from the original source rather than from the temporary.

; CHECK-LABEL: @wobble
define void @wobble(i8* noalias %dst, i8* %src, i1 %some_condition) {
bb:
  %temp = alloca i8, i32 64
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %temp, i8* nonnull %src, i64 64, i32 8, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %temp, i8* nonnull %src, i64 64, i32 8, i1 false)
  br i1 %some_condition, label %more, label %out

out:
  call void @qux()
  unreachable

more:
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %temp, i64 64, i32 8, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %dst, i8* %src, i64 64, i32 8, i1 false)
  ret void
}

; A CFG triangle with a partial redundancy targeting an alloca. Test that the
; memcpy inside the triangle is optimized to copy directly from the original
; source rather than from the temporary.

; CHECK-LABEL: @foo
define i32 @foo(i1 %t3) {
bb:
  %s = alloca %struct.s, align 4
  %t = alloca %struct.s, align 4
  %s1 = bitcast %struct.s* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %s1, i8* bitcast (%struct.s* @s_foo to i8*), i64 8, i32 4, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %s1, i8* bitcast (%struct.s* @s_foo to i8*), i64 8, i32 4, i1 false)
  br i1 %t3, label %bb4, label %bb7

bb4:                                              ; preds = %bb
  %t5 = bitcast %struct.s* %t to i8*
  %s6 = bitcast %struct.s* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t5, i8* %s6, i64 8, i32 4, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t5, i8* bitcast (%struct.s* @s_foo to i8*), i64 8, i32 4, i1 false)
  br label %bb7

bb7:                                              ; preds = %bb4, %bb
  %t8 = getelementptr %struct.s, %struct.s* %t, i32 0, i32 0
  %t9 = load i32, i32* %t8, align 4
  %t10 = getelementptr %struct.s, %struct.s* %t, i32 0, i32 1
  %t11 = load i32, i32* %t10, align 4
  %t12 = add i32 %t9, %t11
  ret i32 %t12
}

; A CFG diamond with an invoke on one side, and a partially redundant memcpy
; into an alloca on the other. Test that the memcpy inside the diamond is
; optimized to copy ; directly from the original source rather than from the
; temporary. This more complex test represents a relatively common usage
; pattern.

; CHECK-LABEL: @baz
define i32 @baz(i1 %t5) personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*) {
bb:
  %s = alloca %struct.s, align 4
  %t = alloca %struct.s, align 4
  %s3 = bitcast %struct.s* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %s3, i8* bitcast (%struct.s* @s_baz to i8*), i64 8, i32 4, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %s3, i8* bitcast (%struct.s* @s_baz to i8*), i64 8, i32 4, i1 false)
  br i1 %t5, label %bb6, label %bb22

bb6:                                              ; preds = %bb
  invoke void @__cxa_throw(i8* null, i8* bitcast (i8** @i to i8*), i8* null)
          to label %bb25 unwind label %bb9

bb9:                                              ; preds = %bb6
  %t10 = landingpad { i8*, i32 }
          catch i8* null
  br label %bb13

bb13:                                             ; preds = %bb9
  %t15 = call i8* @__cxa_begin_catch(i8* null)
  br label %bb23

bb22:                                             ; preds = %bb
  %t23 = bitcast %struct.s* %t to i8*
  %s24 = bitcast %struct.s* %s to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t23, i8* %s24, i64 8, i32 4, i1 false)
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %t23, i8* bitcast (%struct.s* @s_baz to i8*), i64 8, i32 4, i1 false)
  br label %bb23

bb23:                                             ; preds = %bb22, %bb13
  %t17 = getelementptr inbounds %struct.s, %struct.s* %t, i32 0, i32 0
  %t18 = load i32, i32* %t17, align 4
  %t19 = getelementptr inbounds %struct.s, %struct.s* %t, i32 0, i32 1
  %t20 = load i32, i32* %t19, align 4
  %t21 = add nsw i32 %t18, %t20
  ret i32 %t21

bb25:                                             ; preds = %bb6
  unreachable
}
