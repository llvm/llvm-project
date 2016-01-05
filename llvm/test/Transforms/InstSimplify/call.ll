; RUN: opt < %s -instsimplify -S | FileCheck %s

declare {i8, i1} @llvm.uadd.with.overflow.i8(i8 %a, i8 %b)
declare {i8, i1} @llvm.usub.with.overflow.i8(i8 %a, i8 %b)
declare {i8, i1} @llvm.ssub.with.overflow.i8(i8 %a, i8 %b)
declare {i8, i1} @llvm.umul.with.overflow.i8(i8 %a, i8 %b)

define i1 @test_uadd1() {
; CHECK-LABEL: @test_uadd1(
  %x = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 254, i8 3)
  %overflow = extractvalue {i8, i1} %x, 1
  ret i1 %overflow
; CHECK-NEXT: ret i1 true
}

define i8 @test_uadd2() {
; CHECK-LABEL: @test_uadd2(
  %x = call {i8, i1} @llvm.uadd.with.overflow.i8(i8 254, i8 44)
  %result = extractvalue {i8, i1} %x, 0
  ret i8 %result
; CHECK-NEXT: ret i8 42
}

define {i8, i1} @test_usub1(i8 %V) {
; CHECK-LABEL: @test_usub1(
  %x = call {i8, i1} @llvm.usub.with.overflow.i8(i8 %V, i8 %V)
  ret {i8, i1} %x
; CHECK-NEXT: ret { i8, i1 } zeroinitializer
}

define {i8, i1} @test_ssub1(i8 %V) {
; CHECK-LABEL: @test_ssub1(
  %x = call {i8, i1} @llvm.ssub.with.overflow.i8(i8 %V, i8 %V)
  ret {i8, i1} %x
; CHECK-NEXT: ret { i8, i1 } zeroinitializer
}

define {i8, i1} @test_umul1(i8 %V) {
; CHECK-LABEL: @test_umul1(
  %x = call {i8, i1} @llvm.umul.with.overflow.i8(i8 %V, i8 0)
  ret {i8, i1} %x
; CHECK-NEXT: ret { i8, i1 } zeroinitializer
}

declare i256 @llvm.cttz.i256(i256 %src, i1 %is_zero_undef)

define i256 @test_cttz() {
; CHECK-LABEL: @test_cttz(
  %x = call i256 @llvm.cttz.i256(i256 10, i1 false)
  ret i256 %x
; CHECK-NEXT: ret i256 1
}

declare i256 @llvm.ctpop.i256(i256 %src)

define i256 @test_ctpop() {
; CHECK-LABEL: @test_ctpop(
  %x = call i256 @llvm.ctpop.i256(i256 10)
  ret i256 %x
; CHECK-NEXT: ret i256 2
}

; Test a non-intrinsic that we know about as a library call.
declare float @fabs(float %x)

define float @test_fabs_libcall() {
; CHECK-LABEL: @test_fabs_libcall(

  %x = call float @fabs(float -42.0)
; This is still a real function call, so instsimplify won't nuke it -- other
; passes have to do that.
; CHECK-NEXT: call float @fabs

  ret float %x
; CHECK-NEXT: ret float 4.2{{0+}}e+01
}


declare float @llvm.fabs.f32(float) nounwind readnone
declare float @llvm.floor.f32(float) nounwind readnone
declare float @llvm.ceil.f32(float) nounwind readnone
declare float @llvm.trunc.f32(float) nounwind readnone
declare float @llvm.rint.f32(float) nounwind readnone
declare float @llvm.nearbyint.f32(float) nounwind readnone

; Test idempotent intrinsics
define float @test_idempotence(float %a) {
; CHECK-LABEL: @test_idempotence(

; CHECK: fabs
; CHECK-NOT: fabs
  %a0 = call float @llvm.fabs.f32(float %a)
  %a1 = call float @llvm.fabs.f32(float %a0)

; CHECK: floor
; CHECK-NOT: floor
  %b0 = call float @llvm.floor.f32(float %a)
  %b1 = call float @llvm.floor.f32(float %b0)

; CHECK: ceil
; CHECK-NOT: ceil
  %c0 = call float @llvm.ceil.f32(float %a)
  %c1 = call float @llvm.ceil.f32(float %c0)

; CHECK: trunc
; CHECK-NOT: trunc
  %d0 = call float @llvm.trunc.f32(float %a)
  %d1 = call float @llvm.trunc.f32(float %d0)

; CHECK: rint
; CHECK-NOT: rint
  %e0 = call float @llvm.rint.f32(float %a)
  %e1 = call float @llvm.rint.f32(float %e0)

; CHECK: nearbyint
; CHECK-NOT: nearbyint
  %f0 = call float @llvm.nearbyint.f32(float %a)
  %f1 = call float @llvm.nearbyint.f32(float %f0)

  %r0 = fadd float %a1, %b1
  %r1 = fadd float %r0, %c1
  %r2 = fadd float %r1, %d1
  %r3 = fadd float %r2, %e1
  %r4 = fadd float %r3, %f1

  ret float %r4
}

define i8* @operator_new() {
entry:
  %call = tail call noalias i8* @_Znwm(i64 8)
  %cmp = icmp eq i8* %call, null
  br i1 %cmp, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %add.ptr = getelementptr inbounds i8, i8* %call, i64 4
  br label %cast.end

cast.end:                                         ; preds = %cast.notnull, %entry
  %cast.result = phi i8* [ %add.ptr, %cast.notnull ], [ null, %entry ]
  ret i8* %cast.result

; CHECK-LABEL: @operator_new
; CHECK: br i1 false, label %cast.end, label %cast.notnull
}

declare nonnull noalias i8* @_Znwm(i64)

%"struct.std::nothrow_t" = type { i8 }
@_ZSt7nothrow = external global %"struct.std::nothrow_t"

define i8* @operator_new_nothrow_t() {
entry:
  %call = tail call noalias i8* @_ZnamRKSt9nothrow_t(i64 8, %"struct.std::nothrow_t"* @_ZSt7nothrow)
  %cmp = icmp eq i8* %call, null
  br i1 %cmp, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %add.ptr = getelementptr inbounds i8, i8* %call, i64 4
  br label %cast.end

cast.end:                                         ; preds = %cast.notnull, %entry
  %cast.result = phi i8* [ %add.ptr, %cast.notnull ], [ null, %entry ]
  ret i8* %cast.result

; CHECK-LABEL: @operator_new_nothrow_t
; CHECK: br i1 %cmp, label %cast.end, label %cast.notnull
}

declare i8* @_ZnamRKSt9nothrow_t(i64, %"struct.std::nothrow_t"*) nounwind

define i8* @malloc_can_return_null() {
entry:
  %call = tail call noalias i8* @malloc(i64 8)
  %cmp = icmp eq i8* %call, null
  br i1 %cmp, label %cast.end, label %cast.notnull

cast.notnull:                                     ; preds = %entry
  %add.ptr = getelementptr inbounds i8, i8* %call, i64 4
  br label %cast.end

cast.end:                                         ; preds = %cast.notnull, %entry
  %cast.result = phi i8* [ %add.ptr, %cast.notnull ], [ null, %entry ]
  ret i8* %cast.result

; CHECK-LABEL: @malloc_can_return_null
; CHECK: br i1 %cmp, label %cast.end, label %cast.notnull
}

declare noalias i8* @malloc(i64)
