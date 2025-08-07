; RUN: llvm-ir2vec triplets %s | FileCheck %s -check-prefix=TRIPLETS

define i32 @simple_add(i32 %a, i32 %b) {
entry:
  %add = add i32 %a, %b
  ret i32 %add
}

define i32 @simple_mul(i32 %x, i32 %y) {
entry:
  %mul = mul i32 %x, %y
  ret i32 %mul
}

define i32 @test_function(i32 %arg1, i32 %arg2) {
entry:
  %local1 = alloca i32, align 4
  %local2 = alloca i32, align 4
  store i32 %arg1, ptr %local1, align 4
  store i32 %arg2, ptr %local2, align 4
  %load1 = load i32, ptr %local1, align 4
  %load2 = load i32, ptr %local2, align 4
  %result = add i32 %load1, %load2
  ret i32 %result
}

; TRIPLETS: MAX_RELATION=3
; TRIPLETS-NEXT: 12      79      0
; TRIPLETS-NEXT: 12      91      2
; TRIPLETS-NEXT: 12      91      3
; TRIPLETS-NEXT: 12      0       1
; TRIPLETS-NEXT: 0       74      0
; TRIPLETS-NEXT: 0       91      2
; TRIPLETS-NEXT: 16      79      0
; TRIPLETS-NEXT: 16      91      2
; TRIPLETS-NEXT: 16      91      3
; TRIPLETS-NEXT: 16      0       1
; TRIPLETS-NEXT: 0       74      0
; TRIPLETS-NEXT: 0       91      2
; TRIPLETS-NEXT: 30      81      0
; TRIPLETS-NEXT: 30      90      2
; TRIPLETS-NEXT: 30      30      1
; TRIPLETS-NEXT: 30      81      0
; TRIPLETS-NEXT: 30      90      2
; TRIPLETS-NEXT: 30      32      1
; TRIPLETS-NEXT: 32      74      0
; TRIPLETS-NEXT: 32      91      2
; TRIPLETS-NEXT: 32      89      3
; TRIPLETS-NEXT: 32      32      1
; TRIPLETS-NEXT: 32      74      0
; TRIPLETS-NEXT: 32      91      2
; TRIPLETS-NEXT: 32      89      3
; TRIPLETS-NEXT: 32      31      1
; TRIPLETS-NEXT: 31      79      0
; TRIPLETS-NEXT: 31      89      2
; TRIPLETS-NEXT: 31      31      1
; TRIPLETS-NEXT: 31      79      0
; TRIPLETS-NEXT: 31      89      2
; TRIPLETS-NEXT: 31      12      1
; TRIPLETS-NEXT: 12      79      0
; TRIPLETS-NEXT: 12      91      2
; TRIPLETS-NEXT: 12      91      3
; TRIPLETS-NEXT: 12      0       1
; TRIPLETS-NEXT: 0       74      0
; TRIPLETS-NEXT: 0       91      2
