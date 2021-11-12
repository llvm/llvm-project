; RUN: llvm-reduce %s -o %t --delta-passes=operands-skip --test FileCheck --test-arg %s --test-arg --match-full-lines --test-arg --check-prefix=INTERESTING --test-arg --input-file
; RUN: FileCheck %s --input-file %t --check-prefixes=REDUCED

; INTERESTING: store i32 43, i32* {{(%imm|%indirect)}}, align 4
; REDUCED:     store i32 43, i32* %imm, align 4

; INTERESTING: store i32 44, i32* {{(%imm|%indirect|%phi)}}, align 4
; REDUCED:     store i32 44, i32* %phi, align 4

; INTERESTING: store i32 45, i32* {{(%imm|%indirect|%phi|%val)}}, align 4
; REDUCED:     store i32 45, i32* %val, align 4

; INTERESTING: store i32 46, i32* {{(%imm|%indirect|%phi|%val|@Global)}}, align 4
; REDUCED:     store i32 46, i32* @Global, align 4

; INTERESTING: store i32 47, i32* {{(%imm|%indirect|%phi|%val|@Global|%arg2)}}, align 4
; REDUCED:     store i32 47, i32* %arg2, align 4

; INTERESTING: store i32 48, i32* {{(%imm|%indirect|%phi|%val|@Global|%arg2|%arg1)}}, align 4
; REDUCED:     store i32 48, i32* %arg1, align 4

; INTERESTING: store i32 49, i32* {{(%imm|%indirect|%phi|%val|@Global|%arg2|%arg1|null)}}, align 4
; REDUCED:     store i32 49, i32* null, align 4

; REDUCED:     store i32 50, i32* %arg1, align 4
; REDUCED:     store i32 51, i32* %arg1, align 4

@Global = global i32 42

define void @func(i32* %arg1, i32* %arg2) {
entry:
  %val = getelementptr i32, i32* getelementptr (i32, i32* @Global, i32 1), i32 2
  br i1 undef, label %branch, label %loop

branch:
  %nondominating1 = getelementptr i32, i32* %val, i32 3
  br label %loop

loop:
  %phi = phi i32* [ null, %entry ], [ %nondominating1, %branch ], [ %nondominating2, %loop ]
  %imm = getelementptr i32, i32* %phi, i32 4
  %indirect = getelementptr i32, i32* %imm, i32 5

  store i32 43, i32* %imm, align 4 ; Don't reduce to %indirect (not "more reduced" than %imm)
  store i32 44, i32* %imm, align 4 ; Reduce to %phi
  store i32 45, i32* %imm, align 4 ; Reduce to %val
  store i32 46, i32* %imm, align 4 ; Reduce to @Global
  store i32 47, i32* %imm, align 4 ; Reduce to %arg1
  store i32 48, i32* %imm, align 4 ; Reduce to %arg2
  store i32 49, i32* %imm, align 4 ; Reduce to null

  %nondominating2 = getelementptr i32, i32* %indirect, i32 6
  br i1 undef, label %loop, label %exit

exit:
  store i32 50, i32* %arg2, align 4 ; Reduce to %arg1 (compactify function arguments)
  store i32 51, i32* %arg1, align 4 ; Don't reduce
  ret void
}
