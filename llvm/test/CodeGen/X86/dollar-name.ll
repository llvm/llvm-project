; RUN: llc < %s -march=x86 -mtriple=i386-linux | FileCheck %s
; PR1339

@"$bar" = global i32 zeroinitializer
@"$qux" = external global i32

define i32 @"$foo"() nounwind {
; CHECK: movl	($bar),
; CHECK: addl	($qux),
; CHECK: calll	($hen)
  %m = load i32, i32* @"$bar"
  %n = load i32, i32* @"$qux"
  %t = add i32 %m, %n
  %u = call i32 @"$hen"(i32 %t)
  ret i32 %u
}

declare i32 @"$hen"(i32 %a)
