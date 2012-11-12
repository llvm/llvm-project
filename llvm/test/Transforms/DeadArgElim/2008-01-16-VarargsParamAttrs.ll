; RUN: opt < %s -deadargelim -S | grep byval

	%struct.point = type { double, double }
@pts = global [4 x %struct.point] [ %struct.point { double 1.000000e+00, double 2.000000e+00 }, %struct.point { double 3.000000e+00, double 4.000000e+00 }, %struct.point { double 5.000000e+00, double 6.000000e+00 }, %struct.point { double 7.000000e+00, double 8.000000e+00 } ], align 32		; <[4 x %struct.point]*> [#uses=1]

define internal i32 @va1(i32 %nargs, ...) {
entry:
	%pi = alloca %struct.point		; <%struct.point*> [#uses=0]
	%args = alloca i8*		; <i8**> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%args1 = bitcast i8** %args to i8*		; <i8*> [#uses=1]
	call void @llvm.va_start( i8* %args1 )
	%args41 = bitcast i8** %args to i8*		; <i8*> [#uses=1]
	call void @llvm.va_end( i8* %args41 )
	ret i32 undef
}

declare void @llvm.va_start(i8*) nounwind 

declare void @llvm.va_end(i8*) nounwind 

define i32 @main() {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = getelementptr [4 x %struct.point]* @pts, i32 0, i32 0		; <%struct.point*> [#uses=1]
	%tmp1 = call i32 (i32, ...)* @va1( i32 1, %struct.point* byval  %tmp ) nounwind 		; <i32> [#uses=0]
	call void @exit( i32 0 ) noreturn nounwind 
	unreachable
}

declare void @exit(i32) noreturn nounwind 
