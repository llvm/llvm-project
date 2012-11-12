; RUN: llc < %s -march=x86 > %t
; RUN: grep "movb.7(%...)" %t
; RUN: not grep leal %t

define i8 @test(i32 *%P) nounwind {
  %Q = getelementptr i32* %P, i32 1
  %R = bitcast i32* %Q to i8*
  %S = load i8* %R
  %T = icmp eq i8 %S, 0
  br i1 %T, label %TB, label %F
TB:
  ret i8 4
F:
  %U = getelementptr i8* %R, i32 3
  %V = load i8* %U
  ret i8 %V
}
