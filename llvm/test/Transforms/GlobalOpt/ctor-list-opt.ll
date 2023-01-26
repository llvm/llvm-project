; RUN: opt < %s -passes=globalopt -S | FileCheck %s
; CHECK-NOT: CTOR
%ini = type { i32, ptr, ptr }
@llvm.global_ctors = appending global [16 x %ini] [
	%ini { i32 65534, ptr @CTOR1, ptr null },
	%ini { i32 65535, ptr @CTOR1, ptr null },
	%ini { i32 65535, ptr @CTOR1, ptr null },
	%ini { i32 65535, ptr @CTOR2, ptr null },
	%ini { i32 65535, ptr @CTOR3, ptr null },
	%ini { i32 65535, ptr @CTOR4, ptr null },
	%ini { i32 65535, ptr @CTOR5, ptr null },
	%ini { i32 65535, ptr @CTOR6, ptr null },
	%ini { i32 65535, ptr @CTOR7, ptr null },
	%ini { i32 65535, ptr @CTOR8, ptr null },
	%ini { i32 65535, ptr @CTOR9, ptr null },
	%ini { i32 65535, ptr @CTOR14,ptr null },
	%ini { i32 65536, ptr @CTOR10_EXTERNAL, ptr null },
	%ini { i32 65536, ptr @CTOR11, ptr null },
	%ini { i32 65537, ptr @CTOR12, ptr null },
	%ini { i32 2147483647, ptr null, ptr null }
]

@G = global i32 0		; <ptr> [#uses=1]
@G2 = global i32 0		; <ptr> [#uses=1]
@G3 = global i32 -123		; <ptr> [#uses=2]
@X = global { i32, [2 x i32] } { i32 0, [2 x i32] [ i32 17, i32 21 ] }		; <ptr> [#uses=2]
@Y = global i32 -1		; <ptr> [#uses=2]
@Z = global i32 123		; <ptr> [#uses=1]
@D = global double 0.000000e+00		; <ptr> [#uses=1]
@CTORGV = internal global i1 false		; <ptr> [#uses=2]
@GA = global i32 0		; <ptr> [#uses=1]
@GA14 = global i32 0		; <ptr> [#uses=1]

define internal void @CTOR1() {
	ret void
}

define internal void @CTOR2() {
	%A = add i32 1, 23		; <i32> [#uses=1]
	store i32 %A, ptr @G
	store i1 true, ptr @CTORGV
	ret void
}

define internal void @CTOR3() {
	%X = or i1 true, false		; <i1> [#uses=1]
	br label %Cont

Cont:		; preds = %0
	br i1 %X, label %S, label %T

S:		; preds = %Cont
	store i32 24, ptr @G2
	ret void

T:		; preds = %Cont
	ret void
}

define internal void @CTOR4() {
	%X = load i32, ptr @G3		; <i32> [#uses=1]
	%Y = add i32 %X, 123		; <i32> [#uses=1]
	store i32 %Y, ptr @G3
	ret void
}

define internal void @CTOR5() {
	%X.2p = getelementptr inbounds { i32, [2 x i32] }, ptr @X, i32 0, i32 1, i32 0		; <ptr> [#uses=2]
	%X.2 = load i32, ptr %X.2p		; <i32> [#uses=1]
	%X.1p = getelementptr inbounds { i32, [2 x i32] }, ptr @X, i32 0, i32 0		; <ptr> [#uses=1]
	store i32 %X.2, ptr %X.1p
	store i32 42, ptr %X.2p
	ret void
}

define internal void @CTOR6() {
	%A = alloca i32		; <ptr> [#uses=2]
	%y = load i32, ptr @Y		; <i32> [#uses=1]
	store i32 %y, ptr %A
	%Av = load i32, ptr %A		; <i32> [#uses=1]
	%Av1 = add i32 %Av, 1		; <i32> [#uses=1]
	store i32 %Av1, ptr @Y
	ret void
}

define internal void @CTOR7() {
	call void @setto( ptr @Z, i32 0 )
	ret void
}

define void @setto(ptr %P, i32 %V) {
	store i32 %V, ptr %P
	ret void
}

declare double @cos(double)

define internal void @CTOR8() {
	%X = call double @cos( double 0.000000e+00 )		; <double> [#uses=1]
	store double %X, ptr @D
	ret void
}

define i1 @accessor() {
	%V = load i1, ptr @CTORGV		; <i1> [#uses=1]
	ret i1 %V
}

%struct.A = type { i32 }
%struct.B = type { ptr, ptr, [4 x i8] }
@GV1 = global %struct.B zeroinitializer, align 8
@GV2 =  constant [3 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr null]
; CHECK-NOT: CTOR9
define internal void @CTOR9() {
entry:
  %0 = getelementptr inbounds i8, ptr @GV1, i64 16
  store ptr getelementptr inbounds ([3 x ptr], ptr @GV2, i64 1, i64 0), ptr @GV1
  ret void
}

; CHECK: CTOR10_EXTERNAL
declare external void @CTOR10_EXTERNAL();

; CHECK-NOT: CTOR11
define internal void @CTOR11() {
	ret void
}

; CHECK: CTOR12
define internal void @CTOR12() {
	ret void
}

; CHECK-NOT: CTOR13
define internal void @CTOR13() {
  store atomic i32 123, ptr @GA seq_cst, align 4
  ret void
}

; CHECK-NOT: CTOR14
define internal void @CTOR14() {
  %X = load atomic i32, ptr @GA14 seq_cst, align 4
  %Y = add i32 %X, 124
  store i32 %Y, ptr @GA14
  ret void
}

