; RUN: llvm-as < %s | llvm-dis | FileCheck %s

; CHECK: define i32 @f(i64 "foo bar" %0, i64 %1, i64 %2, i64 "xyz" %3) {
define i32 @f(i64 "foo bar", i64, i64, i64 "xyz") {
  ret i32 41
}
