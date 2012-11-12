; RUN: llc < %s -mattr=+rtm -mtriple=x86_64-unknown-unknown | FileCheck %s

declare i32 @llvm.x86.xbegin() nounwind
declare void @llvm.x86.xend() nounwind
declare void @llvm.x86.xabort(i8) noreturn nounwind

define i32 @test_xbegin() nounwind uwtable {
entry:
  %0 = tail call i32 @llvm.x86.xbegin() nounwind
  ret i32 %0
; CHECK: test_xbegin
; CHECK: xbegin [[LABEL:.*BB.*]]
; CHECK: [[LABEL]]:
}

define void @test_xend() nounwind uwtable {
entry:
  tail call void @llvm.x86.xend() nounwind
  ret void
; CHECK: test_xend
; CHECK: xend
}

define void @test_xabort() nounwind uwtable {
entry:
  tail call void @llvm.x86.xabort(i8 2)
  unreachable
; CHECK: test_xabort
; CHECK: xabort $2
}
