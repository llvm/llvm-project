; RUN: opt < %s -passes='cgscc(devirt<4>(inline))' -S | FileCheck %s
; PR4834

define i32 @test1() {
  %funcall1_ = call fastcc ptr () @f1()
  %executecommandptr1_ = call i32 %funcall1_()
  ret i32 %executecommandptr1_
}

define internal fastcc ptr @f1() nounwind readnone {
  ret ptr @f2
}

define internal i32 @f2() nounwind readnone {
  ret i32 1
}

; CHECK: @test1()
; CHECK-NEXT: ret i32 1





declare ptr @f1a(ptr) ssp align 2

define internal i32 @f2a(ptr %t) inlinehint ssp {
entry:
  ret i32 41
}

define internal i32 @f3a(ptr %__f) ssp {
entry:
  %A = call i32 %__f(ptr undef)
  ret i32 %A
}

define i32 @test2(ptr %this) ssp align 2 {
  %X = call i32 @f3a(ptr @f2a) ssp
  ret i32 %X
}

; CHECK-LABEL: @test2(
; CHECK-NEXT: ret i32 41
