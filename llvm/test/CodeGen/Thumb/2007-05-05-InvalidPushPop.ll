; RUN: llc < %s | not grep r11

target triple = "thumb-unknown-linux-gnueabi"
	%struct.__sched_param = type { i32 }
	%struct.pthread_attr_t = type { i32, i32, %struct.__sched_param, i32, i32, i32, i32, ptr, i32 }
@i.1882 = internal global i32 1		; <ptr> [#uses=2]
@.str = internal constant [14 x i8] c"Thread 1: %d\0A\00"		; <ptr> [#uses=1]
@.str1 = internal constant [14 x i8] c"Thread 2: %d\0A\00"		; <ptr> [#uses=1]

define ptr @f(ptr %a) {
entry:
	%tmp1 = load i32, ptr @i.1882		; <i32> [#uses=1]
	%tmp2 = add i32 %tmp1, 1		; <i32> [#uses=2]
	store i32 %tmp2, ptr @i.1882
	%tmp34 = inttoptr i32 %tmp2 to ptr		; <ptr> [#uses=1]
	ret ptr %tmp34
}

define i32 @main() {
entry:
	%t = alloca i32, align 4		; <ptr> [#uses=4]
	%ret = alloca i32, align 4		; <ptr> [#uses=3]
	%tmp1 = call i32 @pthread_create( ptr %t, ptr null, ptr @f, ptr null )		; <i32> [#uses=0]
	%tmp2 = load i32, ptr %t		; <i32> [#uses=1]
	%tmp4 = call i32 @pthread_join( i32 %tmp2, ptr %ret )		; <i32> [#uses=0]
	%tmp5 = load i32, ptr %ret		; <i32> [#uses=1]
	%tmp7 = call i32 (ptr, ...) @printf( ptr @.str, i32 %tmp5 )		; <i32> [#uses=0]
	%tmp8 = call i32 @pthread_create( ptr %t, ptr null, ptr @f, ptr null )		; <i32> [#uses=0]
	%tmp9 = load i32, ptr %t		; <i32> [#uses=1]
	%tmp11 = call i32 @pthread_join( i32 %tmp9, ptr %ret )		; <i32> [#uses=0]
	%tmp12 = load i32, ptr %ret		; <i32> [#uses=1]
	%tmp14 = call i32 (ptr, ...) @printf( ptr @.str1, i32 %tmp12 )		; <i32> [#uses=0]
	ret i32 0
}

declare i32 @pthread_create(ptr, ptr, ptr, ptr)

declare i32 @pthread_join(i32, ptr)

declare i32 @printf(ptr, ...)
