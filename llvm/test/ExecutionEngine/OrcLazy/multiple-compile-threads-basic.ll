; RUN: lli -jit-kind=orc-lazy -compile-threads=2 -thread-entry hello %s | FileCheck %s
; REQUIRES: thread_support
;
; FIXME: Something hangs here.
; UNSUPPORTED: use_msan_with_origins
;
; CHECK: Hello

@.str = private unnamed_addr constant [7 x i8] c"Hello\0A\00", align 1

define void @hello() {
entry:
  %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([7 x i8], [7 x i8]* @.str, i32 0, i32 0))
  ret void
}

declare i32 @printf(i8*, ...)

define i32 @main(i32 %argc, i8** %argv) {
entry:
  ret i32 0
}
