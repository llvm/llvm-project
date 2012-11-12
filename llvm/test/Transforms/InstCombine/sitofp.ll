; RUN: opt < %s -instcombine -S | not grep itofp

define i1 @test1(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ult double %B, 128.0
  ret i1 %C  ;  True!
}
define i1 @test2(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ugt double %B, -128.1
  ret i1 %C  ;  True!
}

define i1 @test3(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ule double %B, 127.0
  ret i1 %C  ;  true!
}

define i1 @test4(i8 %A) {
  %B = sitofp i8 %A to double
  %C = fcmp ult double %B, 127.0
  ret i1 %C  ;  A != 127
}

define i32 @test5(i32 %A) {
  %B = sitofp i32 %A to double
  %C = fptosi double %B to i32
  %D = uitofp i32 %C to double
  %E = fptoui double %D to i32
  ret i32 %E
}

define i32 @test6(i32 %A) {
	%B = and i32 %A, 7		; <i32> [#uses=1]
	%C = and i32 %A, 32		; <i32> [#uses=1]
	%D = sitofp i32 %B to double		; <double> [#uses=1]
	%E = sitofp i32 %C to double		; <double> [#uses=1]
	%F = fadd double %D, %E		; <double> [#uses=1]
	%G = fptosi double %F to i32		; <i32> [#uses=1]
	ret i32 %G
}

define i32 @test7(i32 %a) nounwind {
	%b = sitofp i32 %a to double		; <double> [#uses=1]
	%c = fptoui double %b to i32		; <i32> [#uses=1]
	ret i32 %c
}

define i32 @test8(i32 %a) nounwind {
	%b = uitofp i32 %a to double		; <double> [#uses=1]
	%c = fptosi double %b to i32		; <i32> [#uses=1]
	ret i32 %c
}

