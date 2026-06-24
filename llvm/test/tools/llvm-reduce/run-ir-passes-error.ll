; RUN: not llvm-reduce --abort-on-invalid-reduction --delta-passes=ir-passes --ir-passes=does-not-parse --test FileCheck --test-arg --check-prefixes=CHECK-INTERESTINGNESS --test-arg %s --test-arg --input-file %s -o /dev/null 2>&1 | FileCheck -check-prefix=ERR %s

; CHECK-INTERESTINGNESS-LABEL: @f1
; ERR: LLVM ERROR: unknown pass name 'does-not-parse'

define i32 @f1(i32 %a) {
  %b = add i32 %a, 5
  %c = add i32 %b, 5
  ret i32 %c
}

define i32 @f2(i32 %a) {
  %b = add i32 %a, 5
  %c = add i32 %b, 5
  ret i32 %c
}

declare void @f3()
