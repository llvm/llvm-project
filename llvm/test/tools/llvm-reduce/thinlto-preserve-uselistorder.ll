; RUN: opt --preserve-bc-uselistorder --thinlto-bc --thinlto-split-lto-unit %s -o %t.0

; RUN: llvm-reduce -j=2 --delta-passes=instructions %t.0 -o %t.1 \
; RUN:     --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py --test-arg llvm-dis --test-arg FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s
; RUN: llvm-dis --preserve-ll-uselistorder %t.1 -o %t.2
; RUN: FileCheck --check-prefixes=RESULT,RESULT-PARALLEL %s < %t.2

; FIXME: The single thread path uses CloneModule, which does not
; preserve uselistorder. Consequently it is incapable of reducing
; anything a case that depends on uselistorder.

; RUN: llvm-reduce -j=1 --delta-passes=instructions %t.0 -o %t.3 \
; RUN:     --test %python --test-arg %p/Inputs/llvm-dis-and-filecheck.py --test-arg llvm-dis --test-arg FileCheck --test-arg --check-prefix=INTERESTING --test-arg %s
; RUN: llvm-dis --preserve-ll-uselistorder %t.3 -o %t.4
; RUN: FileCheck --check-prefixes=RESULT,RESULT-SINGLE %s < %t.4

@gv0 = external global [0 x i8]

; RESULT-LABEL: @func(
; RESULT-PARALLEL-NEXT: %gep0 = getelementptr i8, ptr @gv0, i32 %arg0
; RESULT-PARALLEL-NEXT: %gep1 = getelementptr i8, ptr @gv0, i32 %arg1
; RESULT-PARALLEL-NEXT: ret void

; RESULT-SINGLE: ptr @gv0
; RESULT-SINGLE: ptr @gv0
; RESULT-SINGLE: ptr @gv0
; RESULT-SINGLE: ptr @gv0
; RESULT-SINGLE: ptr @gv0
define void @func(i32 %arg0, i32 %arg1, i32 %arg2, i32 %arg3) {
  %gep0 = getelementptr i8, ptr @gv0, i32 %arg0
  %gep1 = getelementptr i8, ptr @gv0, i32 %arg1
  %gep2 = getelementptr i8, ptr @gv0, i32 %arg2
  %gep3 = getelementptr i8, ptr @gv0, i32 %arg3
  store i32 0, ptr %gep0
  store i32 0, ptr %gep1
  store i32 0, ptr %gep2
  store i32 0, ptr %gep3
  store i32 0, ptr @gv0
  ret void
}

; INTERESTING: uselistorder ptr

; RESULT: uselistorder directives
; RESULT-PARALLEL: uselistorder ptr @gv0, { 1, 0 }
; RESULT-SINGLE: uselistorder ptr @gv0, { 3, 4, 2, 1, 0 }

uselistorder ptr @gv0, { 3, 4, 2, 1, 0 }
