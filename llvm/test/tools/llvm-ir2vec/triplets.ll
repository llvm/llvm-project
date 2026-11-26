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

define void @branch() {
entry:
  br label %exit

exit:
  ret void
}

define i1 @cmp(i32 %a, i32 %b, float %x, float %y) {
entry:
  %c = icmp slt i32 %a, %b
  %d = fcmp oeq float %x, %y
  ret i1 %d
}

; TRIPLETS: MAX_RELATION=4
; TRIPLETS-NEXT: 13 77  0
; TRIPLETS-NEXT: 13 87  2
; TRIPLETS-NEXT: 13 87  3
; TRIPLETS-NEXT: 13 0   1
; TRIPLETS-NEXT: 0  72  0
; TRIPLETS-NEXT: 0  87  2
; TRIPLETS-NEXT: 17 77  0
; TRIPLETS-NEXT: 17 87  2
; TRIPLETS-NEXT: 17 87  3
; TRIPLETS-NEXT: 17 0   1
; TRIPLETS-NEXT: 0  72  0
; TRIPLETS-NEXT: 0  87  2
; TRIPLETS-NEXT: 31 80  0
; TRIPLETS-NEXT: 31 86  2
; TRIPLETS-NEXT: 31 31  1
; TRIPLETS-NEXT: 31 80  0
; TRIPLETS-NEXT: 31 86  2
; TRIPLETS-NEXT: 31 33  1
; TRIPLETS-NEXT: 33 72  0
; TRIPLETS-NEXT: 33 87  2
; TRIPLETS-NEXT: 33 85  3
; TRIPLETS-NEXT: 33 33  1
; TRIPLETS-NEXT: 33 72  0
; TRIPLETS-NEXT: 33 87  2
; TRIPLETS-NEXT: 33 85  3
; TRIPLETS-NEXT: 33 32  1
; TRIPLETS-NEXT: 32 77  0
; TRIPLETS-NEXT: 32 85  2
; TRIPLETS-NEXT: 32 32  1
; TRIPLETS-NEXT: 32 77  0
; TRIPLETS-NEXT: 32 85  2
; TRIPLETS-NEXT: 32 13  1
; TRIPLETS-NEXT: 13 77  0
; TRIPLETS-NEXT: 13 87  2
; TRIPLETS-NEXT: 13 87  3
; TRIPLETS-NEXT: 13 0   1
; TRIPLETS-NEXT: 0  72  0
; TRIPLETS-NEXT: 0  87  2
; TRIPLETS-NEXT: 1  72  0
; TRIPLETS-NEXT: 1  88  2
; TRIPLETS-NEXT: 1  0   1
; TRIPLETS-NEXT: 0  72  0
; TRIPLETS-NEXT: 54 77  0
; TRIPLETS-NEXT: 54 87  2
; TRIPLETS-NEXT: 54 87  3
; TRIPLETS-NEXT: 54 113 4
; TRIPLETS-NEXT: 54 55  1
; TRIPLETS-NEXT: 55 77  0
; TRIPLETS-NEXT: 55 87  2
; TRIPLETS-NEXT: 55 87  3
; TRIPLETS-NEXT: 55 90  4
; TRIPLETS-NEXT: 55 0   1
; TRIPLETS-NEXT: 0  72  0
; TRIPLETS-NEXT: 0  87  2
