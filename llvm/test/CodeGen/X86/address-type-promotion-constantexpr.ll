; RUN: llc < %s -mtriple=x86_64-pc-linux

; PR20314 is a crashing bug. This program does nothing with the load, so just check that the return is 0.

@c = common global [2 x i32] zeroinitializer, align 4
@a = common global i32 0, align 4
@b = internal unnamed_addr constant [2 x i8] c"\01\00", align 1

; CHECK-LABEL: main
; CHECK: xor %eax, %eax
define i32 @main() {
entry:
  %foo = load i8, ptr getelementptr ([2 x i8], ptr @b, i64 0, i64 sext (i8 or (i8 zext (i1 icmp eq (ptr getelementptr inbounds ([2 x i32], ptr @c, i64 0, i64 1), ptr @a) to i8), i8 1) to i64)), align 1
  ret i32 0
}

