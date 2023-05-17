; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu | FileCheck %s
@tm_nest_level = internal thread_local global i32 0
define i64 @z() nounwind {
; CHECK:      movq    $tm_nest_level@TPOFF, %r[[R0:[abcd]]]x
; CHECK-NEXT: addl    %fs:0, %e[[R0]]x
; CHECK-NEXT: andl    $100, %e[[R0]]x

  ret i64 and (i64 ptrtoint (ptr @tm_nest_level to i64), i64 100)
}
