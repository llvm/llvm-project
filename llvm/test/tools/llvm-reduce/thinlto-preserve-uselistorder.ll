; RUN: opt --thinlto-bc --thinlto-split-lto-unit %s -o %t.0
; RUN: llvm-reduce -write-tmp-files-as-bitcode --delta-passes=instructions %t.0 -o %t.1 \
; RUN:     --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py --test-arg llvm-dis --test-arg FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s
; RUN: llvm-dis --preserve-ll-uselistorder %t.1 -o %t.2
; RUN: FileCheck --check-prefix=RESULT %s < %t.2

define i32 @func(i32 %arg0, i32 %arg1) {
entry:
  %add0 = add i32 %arg0, 0
  %add1 = add i32 %add0, 0
  %add2 = add i32 %add1, 0
  %add3 = add i32 %arg1, 0
  %add4 = add i32 %add2, %add3
  ret i32 %add4
}

; INTERESTING: uselistorder i32 0
; RESULT: uselistorder i32 0, { 0, 2, 1 }
uselistorder i32 0, { 3, 2, 1, 0 }
