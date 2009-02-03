; RUN: llvm-as < %s | llc -march=x86 -mcpu=pentium 
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah 
; RUN: llvm-as < %s | llc -march=x86 -mcpu=yonah  | not grep set

define i1 @boolSel(i1 %A, i1 %B, i1 %C) nounwind {
	%X = select i1 %A, i1 %B, i1 %C		; <i1> [#uses=1]
	ret i1 %X
}

define i8 @byteSel(i1 %A, i8 %B, i8 %C) nounwind {
	%X = select i1 %A, i8 %B, i8 %C		; <i8> [#uses=1]
	ret i8 %X
}

define i16 @shortSel(i1 %A, i16 %B, i16 %C) nounwind {
	%X = select i1 %A, i16 %B, i16 %C		; <i16> [#uses=1]
	ret i16 %X
}

define i32 @intSel(i1 %A, i32 %B, i32 %C) nounwind {
	%X = select i1 %A, i32 %B, i32 %C		; <i32> [#uses=1]
	ret i32 %X
}

define i64 @longSel(i1 %A, i64 %B, i64 %C) nounwind {
	%X = select i1 %A, i64 %B, i64 %C		; <i64> [#uses=1]
	ret i64 %X
}

define double @doubleSel(i1 %A, double %B, double %C) nounwind {
	%X = select i1 %A, double %B, double %C		; <double> [#uses=1]
	ret double %X
}

define i8 @foldSel(i1 %A, i8 %B, i8 %C) nounwind {
	%Cond = icmp slt i8 %B, %C		; <i1> [#uses=1]
	%X = select i1 %Cond, i8 %B, i8 %C		; <i8> [#uses=1]
	ret i8 %X
}

define i32 @foldSel2(i1 %A, i32 %B, i32 %C) nounwind {
	%Cond = icmp eq i32 %B, %C		; <i1> [#uses=1]
	%X = select i1 %Cond, i32 %B, i32 %C		; <i32> [#uses=1]
	ret i32 %X
}

define i32 @foldSel2a(i1 %A, i32 %B, i32 %C, double %X, double %Y) nounwind {
	%Cond = fcmp olt double %X, %Y		; <i1> [#uses=1]
	%X.upgrd.1 = select i1 %Cond, i32 %B, i32 %C		; <i32> [#uses=1]
	ret i32 %X.upgrd.1
}

define float @foldSel3(i1 %A, float %B, float %C, i32 %X, i32 %Y) nounwind {
	%Cond = icmp ult i32 %X, %Y		; <i1> [#uses=1]
	%X.upgrd.2 = select i1 %Cond, float %B, float %C		; <float> [#uses=1]
	ret float %X.upgrd.2
}

define float @nofoldSel4(i1 %A, float %B, float %C, i32 %X, i32 %Y) nounwind {
	%Cond = icmp slt i32 %X, %Y		; <i1> [#uses=1]
	%X.upgrd.3 = select i1 %Cond, float %B, float %C		; <float> [#uses=1]
	ret float %X.upgrd.3
}
