; RUN: opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK
; RUN: opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_5D_vocab.json %s 2>&1 | FileCheck %s -check-prefix=5D-CHECK

define dso_local i32 @abc(i32 %0, i32 %1) {
entry:
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = load i32, ptr %3, align 4
  %8 = mul nsw i32 %6, %7
  %9 = add nsw i32 %5, %8
  ret i32 %9
}

; 3D-CHECK: IR2Vec embeddings for function abc:
; 3D-CHECK-NEXT: Function vector:  [ 51.00 60.00 69.00 ]
; 3D-CHECK-NEXT: Basic block vectors:
; 3D-CHECK-NEXT: Basic block: entry:
; 3D-CHECK-NEXT:  [ 51.00 60.00 69.00 ]
; 3D-CHECK-NEXT: Instruction vectors:
; 3D-CHECK-NEXT: Instruction:   %2 = alloca i32, align 4 [ 1.00 2.00 3.00 ]
; 3D-CHECK-NEXT: Instruction:   %3 = alloca i32, align 4 [ 1.00 2.00 3.00 ]
; 3D-CHECK-NEXT: Instruction:   store i32 %0, ptr %2, align 4 [ 7.00 8.00 9.00 ]
; 3D-CHECK-NEXT: Instruction:   store i32 %1, ptr %3, align 4 [ 7.00 8.00 9.00 ]
; 3D-CHECK-NEXT: Instruction:   %4 = load i32, ptr %2, align 4 [ 4.00 5.00 6.00 ]
; 3D-CHECK-NEXT: Instruction:   %5 = load i32, ptr %3, align 4 [ 4.00 5.00 6.00 ]
; 3D-CHECK-NEXT: Instruction:   %6 = load i32, ptr %2, align 4 [ 4.00 5.00 6.00 ]
; 3D-CHECK-NEXT: Instruction:   %7 = mul nsw i32 %5, %6 [ 13.00 14.00 15.00 ]
; 3D-CHECK-NEXT: Instruction:   %8 = add nsw i32 %4, %7 [ 10.00 11.00 12.00 ]
; 3D-CHECK-NEXT: Instruction:   ret i32 %8 [ 0.00 0.00 0.00 ]

; 5D-CHECK: IR2Vec embeddings for function abc:
; 5D-CHECK-NEXT: Function vector:  [ 16.50  22.00  27.50  61.50  72.95 ]
; 5D-CHECK-NEXT: Basic block vectors:
; 5D-CHECK-NEXT: Basic block: entry:
; 5D-CHECK-NEXT:  [ 16.50  22.00  27.50  61.50  72.95 ]
; 5D-CHECK-NEXT: Instruction vectors:
; 5D-CHECK-NEXT: Instruction:   %2 = alloca i32, align 4 [ -0.10  -0.20  -0.30  1.00  2.00 ]
; 5D-CHECK-NEXT: Instruction:   %3 = alloca i32, align 4 [ -0.10  -0.20  -0.30  1.00  2.00 ]
; 5D-CHECK-NEXT: Instruction:   store i32 %0, ptr %2, align 4 [ -0.30  0.20  0.70  9.20  10.80 ]
; 5D-CHECK-NEXT: Instruction:   store i32 %1, ptr %3, align 4 [ -0.30  0.20  0.70  9.20  10.80 ]
; 5D-CHECK-NEXT: Instruction:   %4 = load i32, ptr %2, align 4 [ -0.30  -0.10  0.10  5.10  6.05 ]
; 5D-CHECK-NEXT: Instruction:   %5 = load i32, ptr %3, align 4 [ -0.30  -0.10  0.10  5.10  6.05 ]
; 5D-CHECK-NEXT: Instruction:   %6 = load i32, ptr %2, align 4 [ -0.30  -0.10  0.10  5.10  6.05 ]
; 5D-CHECK-NEXT: Instruction:   %7 = mul nsw i32 %5, %6 [ 12.90  14.80  16.70  2.50  2.95 ]
; 5D-CHECK-NEXT: Instruction:   %8 = add nsw i32 %4, %7 [ -0.10  0.70  1.50  13.70  15.25 ]
; 5D-CHECK-NEXT: Instruction:   ret i32 %8 [ 5.40  6.80  8.20  9.60  11.00 ]
