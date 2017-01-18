; RUN: opt < %s -inline -S | FileCheck %s
; RUN: opt < %s -passes='cgscc(inline)' -S | FileCheck %s

; We have to apply the less restrictive TailCallKind of the call site being
; inlined and any call sites cloned into the caller.

; No tail marker after inlining, since test_capture_c captures an alloca.
; CHECK: define void @test_capture_a(
; CHECK-NOT: tail
; CHECK: call void @test_capture_c(

declare void @test_capture_c(i32*)
define internal void @test_capture_b(i32* %P) {
  tail call void @test_capture_c(i32* %P)
  ret void
}
define void @test_capture_a() {
  %A = alloca i32  		; captured by test_capture_b
  call void @test_capture_b(i32* %A)
  ret void
}

; No musttail marker after inlining, since the prototypes don't match.
; CHECK: define void @test_proto_mismatch_a(
; CHECK-NOT: musttail
; CHECK: call void @test_proto_mismatch_c(

declare void @test_proto_mismatch_c(i32*)
define internal void @test_proto_mismatch_b(i32* %p) {
  musttail call void @test_proto_mismatch_c(i32* %p)
  ret void
}
define void @test_proto_mismatch_a() {
  call void @test_proto_mismatch_b(i32* null)
  ret void
}

; After inlining through a musttail call site, we need to keep musttail markers
; to prevent unbounded stack growth.
; CHECK: define void @test_musttail_basic_a(
; CHECK: musttail call void @test_musttail_basic_c(

declare void @test_musttail_basic_c(i32* %p)
define internal void @test_musttail_basic_b(i32* %p) {
  musttail call void @test_musttail_basic_c(i32* %p)
  ret void
}
define void @test_musttail_basic_a(i32* %p) {
  musttail call void @test_musttail_basic_b(i32* %p)
  ret void
}

; Don't insert lifetime end markers here, the lifetime is trivially over due
; the return.
; CHECK: define void @test_byval_a(
; CHECK: musttail call void @test_byval_c(
; CHECK-NEXT: ret void

declare void @test_byval_c(i32* byval %p)
define internal void @test_byval_b(i32* byval %p) {
  musttail call void @test_byval_c(i32* byval %p)
  ret void
}
define void @test_byval_a(i32* byval %p) {
  musttail call void @test_byval_b(i32* byval %p)
  ret void
}

; Don't insert a stack restore, we're about to return.
; CHECK: define void @test_dynalloca_a(
; CHECK: call i8* @llvm.stacksave(
; CHECK: alloca i8, i32 %n
; CHECK: musttail call void @test_dynalloca_c(
; CHECK-NEXT: ret void

declare void @escape(i8* %buf)
declare void @test_dynalloca_c(i32* byval %p, i32 %n)
define internal void @test_dynalloca_b(i32* byval %p, i32 %n) alwaysinline {
  %buf = alloca i8, i32 %n              ; dynamic alloca
  call void @escape(i8* %buf)           ; escape it
  musttail call void @test_dynalloca_c(i32* byval %p, i32 %n)
  ret void
}
define void @test_dynalloca_a(i32* byval %p, i32 %n) {
  musttail call void @test_dynalloca_b(i32* byval %p, i32 %n)
  ret void
}

; We can't merge the returns.
; CHECK: define void @test_multiret_a(
; CHECK: musttail call void @test_multiret_c(
; CHECK-NEXT: ret void
; CHECK: musttail call void @test_multiret_d(
; CHECK-NEXT: ret void

declare void @test_multiret_c(i1 zeroext %b)
declare void @test_multiret_d(i1 zeroext %b)
define internal void @test_multiret_b(i1 zeroext %b) {
  br i1 %b, label %c, label %d
c:
  musttail call void @test_multiret_c(i1 zeroext %b)
  ret void
d:
  musttail call void @test_multiret_d(i1 zeroext %b)
  ret void
}
define void @test_multiret_a(i1 zeroext %b) {
  musttail call void @test_multiret_b(i1 zeroext %b)
  ret void
}

; We have to avoid bitcast chains.
; CHECK: define i32* @test_retptr_a(
; CHECK: musttail call i8* @test_retptr_c(
; CHECK-NEXT: bitcast i8* {{.*}} to i32*
; CHECK-NEXT: ret i32*

declare i8* @test_retptr_c()
define internal i16* @test_retptr_b() {
  %rv = musttail call i8* @test_retptr_c()
  %v = bitcast i8* %rv to i16*
  ret i16* %v
}
define i32* @test_retptr_a() {
  %rv = musttail call i16* @test_retptr_b()
  %v = bitcast i16* %rv to i32*
  ret i32* %v
}

; Combine the last two cases: multiple returns with pointer bitcasts.
; CHECK: define i32* @test_multiptrret_a(
; CHECK: musttail call i8* @test_multiptrret_c(
; CHECK-NEXT: bitcast i8* {{.*}} to i32*
; CHECK-NEXT: ret i32*
; CHECK: musttail call i8* @test_multiptrret_d(
; CHECK-NEXT: bitcast i8* {{.*}} to i32*
; CHECK-NEXT: ret i32*

declare i8* @test_multiptrret_c(i1 zeroext %b)
declare i8* @test_multiptrret_d(i1 zeroext %b)
define internal i16* @test_multiptrret_b(i1 zeroext %b) {
  br i1 %b, label %c, label %d
c:
  %c_rv = musttail call i8* @test_multiptrret_c(i1 zeroext %b)
  %c_v = bitcast i8* %c_rv to i16*
  ret i16* %c_v
d:
  %d_rv = musttail call i8* @test_multiptrret_d(i1 zeroext %b)
  %d_v = bitcast i8* %d_rv to i16*
  ret i16* %d_v
}
define i32* @test_multiptrret_a(i1 zeroext %b) {
  %rv = musttail call i16* @test_multiptrret_b(i1 zeroext %b)
  %v = bitcast i16* %rv to i32*
  ret i32* %v
}

; Inline a musttail call site which contains a normal return and a musttail call.
; CHECK: define i32 @test_mixedret_a(
; CHECK: br i1 %b
; CHECK: musttail call i32 @test_mixedret_c(
; CHECK-NEXT: ret i32
; CHECK: call i32 @test_mixedret_d(i1 zeroext %b)
; CHECK: add i32 1,
; CHECK-NOT: br
; CHECK: ret i32

declare i32 @test_mixedret_c(i1 zeroext %b)
declare i32 @test_mixedret_d(i1 zeroext %b)
define internal i32 @test_mixedret_b(i1 zeroext %b) {
  br i1 %b, label %c, label %d
c:
  %c_rv = musttail call i32 @test_mixedret_c(i1 zeroext %b)
  ret i32 %c_rv
d:
  %d_rv = call i32 @test_mixedret_d(i1 zeroext %b)
  %d_rv1 = add i32 1, %d_rv
  ret i32 %d_rv1
}
define i32 @test_mixedret_a(i1 zeroext %b) {
  %rv = musttail call i32 @test_mixedret_b(i1 zeroext %b)
  ret i32 %rv
}
