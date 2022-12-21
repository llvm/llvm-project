; RUN: llc < %s

@.str_1 = internal constant [7 x i8] c"hello\0A\00"             ; <ptr> [#uses=1]

declare i32 @printf(ptr, ...)

define i32 @main() {
        %s = getelementptr [7 x i8], ptr @.str_1, i64 0, i64 0              ; <ptr> [#uses=1]
        call i32 (ptr, ...) @printf( ptr %s )          ; <i32>:1 [#uses=0]
        ret i32 0
}
