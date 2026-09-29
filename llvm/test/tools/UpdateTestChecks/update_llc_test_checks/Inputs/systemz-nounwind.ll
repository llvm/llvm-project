; RUN: llc -mtriple=s390x-unknown-linux < %s | FileCheck %s

define i32 @unwind(i32 %a, i32 %b) {
  %add = add i32 %a, %b
  ret i32 %add
}

define i32 @nounwind(i32 %a, i32 %b) nounwind {
  %add = add i32 %a, %b
  ret i32 %add
}
