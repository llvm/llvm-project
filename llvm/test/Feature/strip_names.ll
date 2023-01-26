; RUN: opt < %s -S | FileCheck %s
; RUN: opt < %s  | opt -S -discard-value-names | FileCheck --check-prefix=NONAME %s


; CHECK: @GlobalValueName
; CHECK: @foo(i32 %in)
; CHECK: somelabel:
; CHECK:  %GV = load i32, ptr @GlobalValueName
; CHECK:  %add = add i32 %in, %GV
; CHECK:  ret i32 %add

; NONAME: @GlobalValueName
; NONAME: @foo(i32 %0)
; NONAME-NOT: somelabel:
; NONAME:  %2 = load i32, ptr @GlobalValueName
; NONAME:  %3 = add i32 %0, %2
; NONAME:  ret i32 %3

@GlobalValueName = global i32 0

define i32 @foo(i32 %in) {
somelabel:
  %GV = load i32, ptr @GlobalValueName
  %add = add i32 %in, %GV
  ret i32 %add
}
