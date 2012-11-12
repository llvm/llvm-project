; RUN: opt < %s -inline -S | FileCheck %s
; PR4834

define i32 @test1() {
  %funcall1_ = call fastcc i32 ()* ()* @f1()
  %executecommandptr1_ = call i32 %funcall1_()
  ret i32 %executecommandptr1_
}

define internal fastcc i32 ()* @f1() nounwind readnone {
  ret i32 ()* @f2
}

define internal i32 @f2() nounwind readnone {
  ret i32 1
}

; CHECK: @test1()
; CHECK-NEXT: ret i32 1





declare i8* @f1a(i8*) ssp align 2

define internal i32 @f2a(i8* %t) inlinehint ssp {
entry:
  ret i32 41
}

define internal i32 @f3a(i32 (i8*)* %__f) ssp {
entry:
  %A = call i32 %__f(i8* undef)
  ret i32 %A
}

define i32 @test2(i8* %this) ssp align 2 {
  %X = call i32 @f3a(i32 (i8*)* @f2a) ssp
  ret i32 %X
}

; CHECK: @test2
; CHECK-NEXT: ret i32 41
