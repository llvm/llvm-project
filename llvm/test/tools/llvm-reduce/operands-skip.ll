; RUN: llvm-reduce %s -o %t --abort-on-invalid-reduction --delta-passes=operands-skip --test FileCheck --test-arg %s --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=REDUCED

; INTERESTING: store i32 43, ptr {{(%imm|%indirect)}}, align 4
; REDUCED:     store i32 43, ptr %imm, align 4

; INTERESTING: store i32 44, ptr {{(%imm|%indirect|%phi)}}, align 4
; REDUCED:     store i32 44, ptr %phi, align 4

; INTERESTING: store i32 45, ptr {{(%imm|%indirect|%phi|%val)}}, align 4
; REDUCED:     store i32 45, ptr %val, align 4

; INTERESTING: store i32 46, ptr {{(%imm|%indirect|%phi|%val|@Global)}}, align 4
; REDUCED:     store i32 46, ptr @Global, align 4

; INTERESTING: store i32 47, ptr {{(%imm|%indirect|%phi|%val|@Global|%arg2)}}, align 4
; REDUCED:     store i32 47, ptr %arg2, align 4

; INTERESTING: store i32 48, ptr {{(%imm|%indirect|%phi|%val|@Global|%arg2|%arg1)}}, align 4
; REDUCED:     store i32 48, ptr %arg1, align 4

; INTERESTING: store i32 49, ptr {{(%imm|%indirect|%phi|%val|@Global|%arg2|%arg1|null)}}, align 4
; REDUCED:     store i32 49, ptr null, align 4

; REDUCED:     store i32 50, ptr %arg1, align 4
; REDUCED:     store i32 51, ptr %arg1, align 4

@Global = global i32 42

define void @func(ptr %arg1, ptr %arg2) {
entry:
  %val = getelementptr i32, ptr getelementptr (i32, ptr @Global, i32 1), i32 2
  br i1 undef, label %branch, label %loop

branch:
  %nondominating1 = getelementptr i32, ptr %val, i32 3
  br label %loop

loop:
  %phi = phi ptr [ null, %entry ], [ %nondominating1, %branch ], [ %nondominating2, %loop ]
  %imm = getelementptr i32, ptr %phi, i32 4
  %indirect = getelementptr i32, ptr %imm, i32 5

  store i32 43, ptr %imm, align 4 ; Don't reduce to %indirect (not "more reduced" than %imm)
  store i32 44, ptr %imm, align 4 ; Reduce to %phi
  store i32 45, ptr %imm, align 4 ; Reduce to %val
  store i32 46, ptr %imm, align 4 ; Reduce to @Global
  store i32 47, ptr %imm, align 4 ; Reduce to %arg1
  store i32 48, ptr %imm, align 4 ; Reduce to %arg2
  store i32 49, ptr %imm, align 4 ; Reduce to null

  %nondominating2 = getelementptr i32, ptr %indirect, i32 6
  br i1 undef, label %loop, label %exit

exit:
  store i32 50, ptr %arg2, align 4 ; Reduce to %arg1 (compactify function arguments)
  store i32 51, ptr %arg1, align 4 ; Don't reduce
  ret void
}
