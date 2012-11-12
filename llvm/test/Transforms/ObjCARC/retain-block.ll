; RUN: opt -objc-arc -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64"

!0 = metadata !{}

declare i8* @objc_retain(i8*)
declare void @callee(i8)
declare void @use_pointer(i8*)
declare void @objc_release(i8*)
declare i8* @objc_retainBlock(i8*)
declare i8* @objc_autorelease(i8*)

; Basic retainBlock+release elimination.

; CHECK: define void @test0(i8* %tmp) {
; CHECK-NOT: @objc
; CHECK: }
define void @test0(i8* %tmp) {
entry:
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
  tail call void @use_pointer(i8* %tmp2)
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  ret void
}

; Same as test0, but there's no copy_on_escape metadata, so there's no
; optimization possible.

; CHECK: define void @test0_no_metadata(i8* %tmp) {
; CHECK: %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind
; CHECK: tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
; CHECK: }
define void @test0_no_metadata(i8* %tmp) {
entry:
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind
  tail call void @use_pointer(i8* %tmp2)
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  ret void
}

; Same as test0, but the pointer escapes, so there's no
; optimization possible.

; CHECK: define void @test0_escape(i8* %tmp, i8** %z) {
; CHECK: %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
; CHECK: tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
; CHECK: }
define void @test0_escape(i8* %tmp, i8** %z) {
entry:
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
  store i8* %tmp2, i8** %z
  tail call void @use_pointer(i8* %tmp2)
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  ret void
}

; Same as test0_escape, but there's no intervening call.

; CHECK: define void @test0_just_escape(i8* %tmp, i8** %z) {
; CHECK: %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
; CHECK: tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
; CHECK: }
define void @test0_just_escape(i8* %tmp, i8** %z) {
entry:
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
  store i8* %tmp2, i8** %z
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  ret void
}

; Basic nested retainBlock+release elimination.

; CHECK: define void @test1(i8* %tmp) {
; CHECK-NOT: @objc
; CHECK: tail call i8* @objc_retain(i8* %tmp) nounwind
; CHECK-NOT: @objc
; CHECK: tail call void @objc_release(i8* %tmp) nounwind, !clang.imprecise_release !0
; CHECK-NOT: @objc
; CHECK: }
define void @test1(i8* %tmp) {
entry:
  %tmp1 = tail call i8* @objc_retain(i8* %tmp) nounwind
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
  tail call void @use_pointer(i8* %tmp2)
  tail call void @use_pointer(i8* %tmp2)
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  tail call void @objc_release(i8* %tmp) nounwind, !clang.imprecise_release !0
  ret void
}

; Same as test1, but there's no copy_on_escape metadata, so there's no
; retainBlock+release optimization possible. But we can still eliminate
; the outer retain+release.

; CHECK: define void @test1_no_metadata(i8* %tmp) {
; CHECK-NEXT: entry:
; CHECK-NEXT: tail call i8* @objc_retainBlock(i8* %tmp) nounwind
; CHECK-NEXT: @use_pointer(i8* %tmp2)
; CHECK-NEXT: @use_pointer(i8* %tmp2)
; CHECK-NEXT: tail call void @objc_release(i8* %tmp) nounwind, !clang.imprecise_release !0
; CHECK-NOT: @objc
; CHECK: }
define void @test1_no_metadata(i8* %tmp) {
entry:
  %tmp1 = tail call i8* @objc_retain(i8* %tmp) nounwind
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind
  tail call void @use_pointer(i8* %tmp2)
  tail call void @use_pointer(i8* %tmp2)
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  tail call void @objc_release(i8* %tmp) nounwind, !clang.imprecise_release !0
  ret void
}

; Same as test1, but the pointer escapes, so there's no
; retainBlock+release optimization possible. But we can still eliminate
; the outer retain+release

; CHECK: define void @test1_escape(i8* %tmp, i8** %z) {
; CHECK-NEXT: entry:
; CHECK-NEXT: %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
; CHECK-NEXT: store i8* %tmp2, i8** %z
; CHECK-NEXT: @use_pointer(i8* %tmp2)
; CHECK-NEXT: @use_pointer(i8* %tmp2)
; CHECK-NEXT: tail call void @objc_release(i8* %tmp) nounwind, !clang.imprecise_release !0
; CHECK-NOT: @objc
; CHECK: }
define void @test1_escape(i8* %tmp, i8** %z) {
entry:
  %tmp1 = tail call i8* @objc_retain(i8* %tmp) nounwind
  %tmp2 = tail call i8* @objc_retainBlock(i8* %tmp) nounwind, !clang.arc.copy_on_escape !0
  store i8* %tmp2, i8** %z
  tail call void @use_pointer(i8* %tmp2)
  tail call void @use_pointer(i8* %tmp2)
  tail call void @objc_release(i8* %tmp2) nounwind, !clang.imprecise_release !0
  tail call void @objc_release(i8* %tmp) nounwind, !clang.imprecise_release !0
  ret void
}
