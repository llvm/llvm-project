; RUN: llvm-as -o %t.bc %s

; RUN: llvm-reduce -j=1 --abort-on-invalid-reduction \
; RUN:   --delta-passes=instructions -o %t.reduced.bc \
; RUN:   --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py \
; RUN:   --test-arg llvm-dis \
; RUN:   --test-arg FileCheck --test-arg --check-prefix=INTERESTING \
; RUN:   --test-arg %s %t.bc

; RUN: llvm-dis --preserve-ll-uselistorder -o %t.reduced.ll %t.reduced.bc

; RUN: FileCheck -check-prefix=RESULT %s < %t.reduced.ll


; INTERESTING: add
; INTERESTING: add
; INTERESTING: add
define i32 @func(i32 %arg0, i32 %arg1) {
entry:
  %add0 = add i32 %arg0, 0
  %add1 = add i32 %add0, 0
  %add2 = add i32 %add1, 0
  %add3 = add i32 %arg1, 0
  %add4 = add i32 %add2, %add3
  ret i32 %add4
}

; INTERESTING: uselistorder
; RESULT: uselistorder
uselistorder i32 0, { 3, 2, 1, 0 }
