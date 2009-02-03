; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux | grep {(\$bar)} | count 1
; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux | grep {(\$qux)} | count 1
; RUN: llvm-as < %s | llc -march=x86 -mtriple=i386-linux | grep {(\$hen)} | count 1
; PR1339

@"$bar" = global i32 zeroinitializer
@"$qux" = external global i32

define i32 @"$foo"() nounwind {
  %m = load i32* @"$bar"
  %n = load i32* @"$qux"
  %t = add i32 %m, %n
  %u = call i32 @"$hen"(i32 %t)
  ret i32 %u
}

declare i32 @"$hen"(i32 %a)
