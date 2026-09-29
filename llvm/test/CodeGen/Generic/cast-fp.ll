; RUN: llc < %s
@a_fstr = internal constant [8 x i8] c"a = %f\0A\00"		; <ptr> [#uses=1]
@a_lstr = internal constant [10 x i8] c"a = %lld\0A\00"		; <ptr> [#uses=1]
@a_dstr = internal constant [8 x i8] c"a = %d\0A\00"		; <ptr> [#uses=1]
@b_dstr = internal constant [8 x i8] c"b = %d\0A\00"		; <ptr> [#uses=1]
@b_fstr = internal constant [8 x i8] c"b = %f\0A\00"		; <ptr> [#uses=1]
@A = global double 2.000000e+00		; <ptr> [#uses=1]
@B = global i32 2		; <ptr> [#uses=1]

declare i32 @printf(ptr, ...)

define i32 @main() {
	%a = load double, ptr @A		; <double> [#uses=4]
	%a_fs = getelementptr [8 x i8], ptr @a_fstr, i64 0, i64 0		; <ptr> [#uses=1]
	call i32 (ptr, ...) @printf( ptr %a_fs, double %a )		; <i32>:1 [#uses=0]
	%a_d2l = fptosi double %a to i64		; <i64> [#uses=1]
	%a_ls = getelementptr [10 x i8], ptr @a_lstr, i64 0, i64 0		; <ptr> [#uses=1]
	call i32 (ptr, ...) @printf( ptr %a_ls, i64 %a_d2l )		; <i32>:2 [#uses=0]
	%a_d2i = fptosi double %a to i32		; <i32> [#uses=2]
	%a_ds = getelementptr [8 x i8], ptr @a_dstr, i64 0, i64 0		; <ptr> [#uses=3]
	call i32 (ptr, ...) @printf( ptr %a_ds, i32 %a_d2i )		; <i32>:3 [#uses=0]
	%a_d2sb = fptosi double %a to i8		; <i8> [#uses=1]
	call i32 (ptr, ...) @printf( ptr %a_ds, i8 %a_d2sb )		; <i32>:4 [#uses=0]
	%a_d2i2sb = trunc i32 %a_d2i to i8		; <i8> [#uses=1]
	call i32 (ptr, ...) @printf( ptr %a_ds, i8 %a_d2i2sb )		; <i32>:5 [#uses=0]
	%b = load i32, ptr @B		; <i32> [#uses=2]
	%b_ds = getelementptr [8 x i8], ptr @b_dstr, i64 0, i64 0		; <ptr> [#uses=1]
	call i32 (ptr, ...) @printf( ptr %b_ds, i32 %b )		; <i32>:6 [#uses=0]
	%b_i2d = sitofp i32 %b to double		; <double> [#uses=1]
	%b_fs = getelementptr [8 x i8], ptr @b_fstr, i64 0, i64 0		; <ptr> [#uses=1]
	call i32 (ptr, ...) @printf( ptr %b_fs, double %b_i2d )		; <i32>:7 [#uses=0]
	ret i32 0
}
