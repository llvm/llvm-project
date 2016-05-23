; RUN: llc < %s -mattr=+rtm -mtriple=x86_64-unknown-unknown | FileCheck %s

declare i32 @llvm.x86.xbegin() nounwind
declare void @llvm.x86.xend() nounwind
declare void @llvm.x86.xabort(i8) nounwind
declare void @f1()

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
  ret void
; CHECK: test_xabort
; CHECK: xabort $2
}

define void @f2(i32 %x) nounwind uwtable {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.x86.xabort(i8 1)
  call void @f1()
  ret void
; CHECK-LABEL: f2
; CHECK: xabort  $1
; CHECK: callq   f1
}
 