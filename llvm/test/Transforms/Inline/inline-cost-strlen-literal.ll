; RUN: opt -p "print<inline-cost>" -disable-output < %s 2>&1 | FileCheck %s

; CHECK: NumInstructionsSimplified: 8
; CHECK: Cost: 10
; CHECK: Threshold: 337

; bool f(const std::string &s) { return s == "s"; }

@.str = private constant [2 x i8] c"s\00", align 1

@G = global [16 x i8] zeroinitializer

define i1 @f(ptr %s, i64 %n) {
entry:
  %len = call i64 @strlen(ptr @.str)
  %c.0 = icmp eq i64 %n, %len
  br i1 %c.0, label %L0, label %exit

L0:
  %c.1 = icmp eq i64 %n, 0
  br i1 %c.1, label %exit, label %L1

L1:
  %t = call i32 @memcmp(ptr %s, ptr @.str, i64 %n)
  %c.2 = icmp eq i32 %t, 0
  br label %exit

exit:
  %res = phi i1 [ false, %entry ], [true, %L0 ], [ %c.2, %L1 ]
  ret i1 %res
}

define void @g() {
  call i1 @f(ptr @G, i64 1)
  ret void
}

declare i64 @strlen(ptr)

declare i32 @memcmp(ptr, ptr, i64)
