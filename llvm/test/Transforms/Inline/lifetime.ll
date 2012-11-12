; RUN: opt -inline %s -S -o - | FileCheck %s

declare void @llvm.lifetime.start(i64, i8*)
declare void @llvm.lifetime.end(i64, i8*)

define void @helper_both_markers() {
  %a = alloca i8
  call void @llvm.lifetime.start(i64 1, i8* %a)
  call void @llvm.lifetime.end(i64 1, i8* %a)
  ret void
}

define void @test_both_markers() {
; CHECK: @test_both_markers
; CHECK: llvm.lifetime.start(i64 1
; CHECK-NEXT: llvm.lifetime.end(i64 1
  call void @helper_both_markers()
; CHECK-NEXT: llvm.lifetime.start(i64 1
; CHECK-NEXT: llvm.lifetime.end(i64 1
  call void @helper_both_markers()
; CHECK-NEXT: ret void
  ret void
}

;; Without this, the inliner will simplify out @test_no_marker before adding
;; any lifetime markers.
declare void @use(i8* %a)

define void @helper_no_markers() {
  %a = alloca i8
  call void @use(i8* %a)
  ret void
}

;; We can't use CHECK-NEXT because there's an extra call void @use in between.
;; Instead, we use CHECK-NOT to verify that there are no other lifetime calls.
define void @test_no_marker() {
; CHECK: @test_no_marker
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start(i64 -1
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end(i64 -1
  call void @helper_no_markers()
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start(i64 -1
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end(i64 -1
  call void @helper_no_markers()
; CHECK-NOT: lifetime
; CHECK: ret void
  ret void
}

define void @helper_two_casts() {
  %a = alloca i32
  %b = bitcast i32* %a to i8*
  call void @llvm.lifetime.start(i64 4, i8* %b)
  %c = bitcast i32* %a to i8*
  call void @llvm.lifetime.end(i64 4, i8* %c)
  ret void
}

define void @test_two_casts() {
; CHECK: @test_two_casts
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start(i64 4
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end(i64 4
  call void @helper_two_casts()
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.start(i64 4
; CHECK-NOT: lifetime
; CHECK: llvm.lifetime.end(i64 4
  call void @helper_two_casts()
; CHECK-NOT: lifetime
; CHECK: ret void
  ret void
}
