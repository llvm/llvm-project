; RUN: llc -mtriple=x86_64 < %s | FileCheck %s --check-prefix=CHECK
; RUN: llc -mtriple=x86_64 -stop-after=finalize-isel < %s | FileCheck %s --check-prefix=CHECK

define i32 @add(i32 %a, i32 %b) {
  %sum = add i32 %a, %b
  ret i32 %sum
}

define i32 @sub(i32 %a, i32 %b) {
  %diff = sub i32 %a, %b
  ret i32 %diff
}

