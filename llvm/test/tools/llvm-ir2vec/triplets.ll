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
; TRIPLETS-NEXT: 13	75	0
; TRIPLETS-NEXT: 13	84	2
; TRIPLETS-NEXT: 13	84	3
; TRIPLETS-NEXT: 13	0	1
; TRIPLETS-NEXT: 0	70	0
; TRIPLETS-NEXT: 0	84	2
; TRIPLETS-NEXT: 17	75	0
; TRIPLETS-NEXT: 17	84	2
; TRIPLETS-NEXT: 17	84	3
; TRIPLETS-NEXT: 17	0	1
; TRIPLETS-NEXT: 0	70	0
; TRIPLETS-NEXT: 0	84	2
; TRIPLETS-NEXT: 31	77	0
; TRIPLETS-NEXT: 31	83	2
; TRIPLETS-NEXT: 31	31	1
; TRIPLETS-NEXT: 31	77	0
; TRIPLETS-NEXT: 31	83	2
; TRIPLETS-NEXT: 31	33	1
; TRIPLETS-NEXT: 33	70	0
; TRIPLETS-NEXT: 33	84	2
; TRIPLETS-NEXT: 33	82	3
; TRIPLETS-NEXT: 33	33	1
; TRIPLETS-NEXT: 33	70	0
; TRIPLETS-NEXT: 33	84	2
; TRIPLETS-NEXT: 33	82	3
; TRIPLETS-NEXT: 33	32	1
; TRIPLETS-NEXT: 32	75	0
; TRIPLETS-NEXT: 32	82	2
; TRIPLETS-NEXT: 32	32	1
; TRIPLETS-NEXT: 32	75	0
; TRIPLETS-NEXT: 32	82	2
; TRIPLETS-NEXT: 32	13	1
; TRIPLETS-NEXT: 13	75	0
; TRIPLETS-NEXT: 13	84	2
; TRIPLETS-NEXT: 13	84	3
; TRIPLETS-NEXT: 13	0	1
; TRIPLETS-NEXT: 0	70	0
; TRIPLETS-NEXT: 0	84	2
