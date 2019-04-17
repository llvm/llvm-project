; RUN: opt < %s -globalopt -S | FileCheck %s
; CHECK-NOT: CTOR
%ini = type { i32, void()*, i8* }
@llvm.global_ctors = appending global [11 x %ini] [
	%ini { i32 65535, void ()* @CTOR1, i8* null },
	%ini { i32 65535, void ()* @CTOR1, i8* null },
	%ini { i32 65535, void ()* @CTOR2, i8* null },
	%ini { i32 65535, void ()* @CTOR3, i8* null },
	%ini { i32 65535, void ()* @CTOR4, i8* null },
	%ini { i32 65535, void ()* @CTOR5, i8* null },
	%ini { i32 65535, void ()* @CTOR6, i8* null },
	%ini { i32 65535, void ()* @CTOR7, i8* null },
	%ini { i32 65535, void ()* @CTOR8, i8* null },
	%ini { i32 65535, void ()* @CTOR9, i8* null },
	%ini { i32 2147483647, void ()* null, i8* null }
]

@G = global i32 0		; <i32*> [#uses=1]
@G2 = global i32 0		; <i32*> [#uses=1]
@G3 = global i32 -123		; <i32*> [#uses=2]
@X = global { i32, [2 x i32] } { i32 0, [2 x i32] [ i32 17, i32 21 ] }		; <{ i32, [2 x i32] }*> [#uses=2]
@Y = global i32 -1		; <i32*> [#uses=2]
@Z = global i32 123		; <i32*> [#uses=1]
@D = global double 0.000000e+00		; <double*> [#uses=1]
@CTORGV = internal global i1 false		; <i1*> [#uses=2]

define internal void @CTOR1() {
	ret void
}

define internal void @CTOR2() {
	%A = add i32 1, 23		; <i32> [#uses=1]
	store i32 %A, i32* @G
	store i1 true, i1* @CTORGV
	ret void
}

define internal void @CTOR3() {
	%X = or i1 true, false		; <i1> [#uses=1]
	br label %Cont

Cont:		; preds = %0
	br i1 %X, label %S, label %T

S:		; preds = %Cont
	store i32 24, i32* @G2
	ret void

T:		; preds = %Cont
	ret void
}

define internal void @CTOR4() {
	%X = load i32, i32* @G3		; <i32> [#uses=1]
	%Y = add i32 %X, 123		; <i32> [#uses=1]
	store i32 %Y, i32* @G3
	ret void
}

define internal void @CTOR5() {
	%X.2p = getelementptr inbounds { i32, [2 x i32] }, { i32, [2 x i32] }* @X, i32 0, i32 1, i32 0		; <i32*> [#uses=2]
	%X.2 = load i32, i32* %X.2p		; <i32> [#uses=1]
	%X.1p = getelementptr inbounds { i32, [2 x i32] }, { i32, [2 x i32] }* @X, i32 0, i32 0		; <i32*> [#uses=1]
	store i32 %X.2, i32* %X.1p
	store i32 42, i32* %X.2p
	ret void
}

define internal void @CTOR6() {
	%A = alloca i32		; <i32*> [#uses=2]
	%y = load i32, i32* @Y		; <i32> [#uses=1]
	store i32 %y, i32* %A
	%Av = load i32, i32* %A		; <i32> [#uses=1]
	%Av1 = add i32 %Av, 1		; <i32> [#uses=1]
	store i32 %Av1, i32* @Y
	ret void
}

define internal void @CTOR7() {
	call void @setto( i32* @Z, i32 0 )
	ret void
}

define void @setto(i32* %P, i32 %V) {
	store i32 %V, i32* %P
	ret void
}

declare double @cos(double)

define internal void @CTOR8() {
	%X = call double @cos( double 0.000000e+00 )		; <double> [#uses=1]
	store double %X, double* @D
	ret void
}

define i1 @accessor() {
	%V = load i1, i1* @CTORGV		; <i1> [#uses=1]
	ret i1 %V
}

%struct.A = type { i32 }
%struct.B = type { i32 (...)**, i8*, [4 x i8] }
@GV1 = global %struct.B zeroinitializer, align 8
@GV2 =  constant [3 x i8*] [i8* inttoptr (i64 16 to i8*), i8* null, i8* bitcast ({ i8*, i8*, i32, i32, i8*, i64 }* null to i8*)]
; CHECK-NOT: CTOR9
define internal void @CTOR9() {
entry:
  %0 = bitcast %struct.B* @GV1 to i8*
  %1 = getelementptr inbounds i8, i8* %0, i64 16
  %2 = bitcast i8* %1 to %struct.A*
  %3 = bitcast %struct.B* @GV1 to i8***
  store i8** getelementptr inbounds ([3 x i8*], [3 x i8*]* @GV2, i64 1, i64 0), i8*** %3
  ret void
}
