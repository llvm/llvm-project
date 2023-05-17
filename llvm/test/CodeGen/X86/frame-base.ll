; RUN: llc -mtriple=x86_64-apple-macosx -o - %s | FileCheck %s

; The issue here was a conflict between forming a %rip-relative lea and a
; FrameIndex lea. The %rip sanity-checks didn't consider that a base register
; had been set if we'd already matched a FrameIndex, when it has in reality.

@var = global i32 0

define void @test_frame_rip_conflict() {
; CHECK-LABEL: test_frame_rip_conflict:
; CHECK: leaq _var(%rip), [[TMPADDR:%r.*]]
; CHECK: leaq {{-?[0-9]+}}(%rsp,[[TMPADDR]]),
  %stackvar = alloca i32

  %stackint = ptrtoint ptr %stackvar to i64
  %addr = add i64 ptrtoint(ptr @var to i64), %stackint

  call void @eat_i64(i64 %addr)
  ret void
}

declare void @eat_i64(i64)
