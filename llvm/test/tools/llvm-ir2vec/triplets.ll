; RUN: llvm-ir2vec --mode=triplets %s | FileCheck %s -check-prefix=TRIPLETS

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

; TRIPLETS: Add IntegerTy Variable Variable
; TRIPLETS-NEXT: Ret VoidTy Variable
; TRIPLETS-NEXT: Mul IntegerTy Variable Variable
; TRIPLETS-NEXT: Ret VoidTy Variable
; TRIPLETS-NEXT: Alloca PointerTy Constant
; TRIPLETS-NEXT: Alloca PointerTy Constant
; TRIPLETS-NEXT: Store VoidTy Variable Pointer
; TRIPLETS-NEXT: Store VoidTy Variable Pointer
; TRIPLETS-NEXT: Load IntegerTy Pointer
; TRIPLETS-NEXT: Load IntegerTy Pointer
; TRIPLETS-NEXT: Add IntegerTy Variable Variable
; TRIPLETS-NEXT: Ret VoidTy Variable
