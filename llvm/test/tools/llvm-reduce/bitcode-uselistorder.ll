; RUN: llvm-as -o %t.bc %s

; RUN: llvm-reduce -j=1 --abort-on-invalid-reduction \
; RUN:   --delta-passes=instructions -o %t.reduced.bc \
; RUN:   --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py \
; RUN:   --test-arg llvm-dis \
; RUN:   --test-arg FileCheck --test-arg --check-prefix=INTERESTING \
; RUN:   --test-arg %s %t.bc

; RUN: llvm-dis --preserve-ll-uselistorder -o %t.reduced.ll %t.reduced.bc

; RUN: FileCheck -check-prefix=RESULT %s < %t.reduced.ll

@gv = external global i32, align 4

; INTERESTING: getelementptr
; INTERESTING: getelementptr
; INTERESTING: getelementptr
define ptr @func(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3, i32 %arg4) {
entry:
  %add0 = getelementptr i8, ptr @gv, i32 %arg0
  %add1 = getelementptr i8, ptr @gv, i32 %arg1
  %add2 = getelementptr i8, ptr @gv, i32 %arg2
  %add3 = getelementptr i8, ptr @gv, i32 %arg3
  %add4 = getelementptr i8, ptr @gv, i32 %arg4
  ret ptr %add4
}

; INTERESTING: uselistorder
; RESULT: uselistorder
uselistorder ptr @gv, { 3, 2, 4, 1, 0 }
