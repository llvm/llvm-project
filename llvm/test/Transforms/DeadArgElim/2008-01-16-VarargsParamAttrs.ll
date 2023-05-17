; RUN: opt < %s -passes=deadargelim -S | grep byval

	%struct.point = type { double, double }
@pts = global [4 x %struct.point] [ %struct.point { double 1.000000e+00, double 2.000000e+00 }, %struct.point { double 3.000000e+00, double 4.000000e+00 }, %struct.point { double 5.000000e+00, double 6.000000e+00 }, %struct.point { double 7.000000e+00, double 8.000000e+00 } ], align 32		; <ptr> [#uses=1]

define internal i32 @va1(i32 %nargs, ...) {
entry:
	%pi = alloca %struct.point		; <ptr> [#uses=0]
	%args = alloca ptr		; <ptr> [#uses=2]
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	call void @llvm.va_start( ptr %args )
	call void @llvm.va_end( ptr %args )
	ret i32 undef
}

declare void @llvm.va_start(ptr) nounwind

declare void @llvm.va_end(ptr) nounwind

define i32 @main() {
entry:
	%"alloca point" = bitcast i32 0 to i32		; <i32> [#uses=0]
	%tmp = getelementptr [4 x %struct.point], ptr @pts, i32 0, i32 0		; <ptr> [#uses=1]
	%tmp1 = call i32 (i32, ...) @va1(i32 1, ptr byval(%struct.point) %tmp) nounwind 		; <i32> [#uses=0]
	call void @exit( i32 0 ) noreturn nounwind
	unreachable
}

declare void @exit(i32) noreturn nounwind
