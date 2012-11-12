; RUN: llc < %s -march=x86 | grep weak | count 2
@__gthrw_pthread_once = alias weak i32 (i32*, void ()*)* @pthread_once		; <i32 (i32*, void ()*)*> [#uses=0]

declare extern_weak i32 @pthread_once(i32*, void ()*)
