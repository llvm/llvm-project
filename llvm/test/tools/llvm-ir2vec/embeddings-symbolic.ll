; RUN: llvm-ir2vec embeddings --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-DEFAULT
; RUN: llvm-ir2vec embeddings --level=func --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-FUNC-LEVEL

; Test with mangled names
; RUN: llvm-ir2vec embeddings --level=func --function=_Z3addii --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-ADD-INT-INT
; RUN: llvm-ir2vec embeddings --level=func --function=_Z3addiii --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-ADD-INT-INT-INT
; RUN: llvm-ir2vec embeddings --level=func --function=_Z3adddd --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-ADD-DOUBLE-DOUBLE
; RUN: llvm-ir2vec embeddings --level=func --function=main --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-MAIN

; Test with demangled names
; RUN: llvm-ir2vec embeddings --level=func --function="add(int, int)" --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-DEMANGLED-INT-INT
; RUN: llvm-ir2vec embeddings --level=func --function="add(int, int, int)" --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-DEMANGLED-INT-INT-INT
; RUN: llvm-ir2vec embeddings --level=func --function="add(double, double)" --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-DEMANGLED-DOUBLE-DOUBLE

; Test basic block level for all functions
; RUN: llvm-ir2vec embeddings --level=bb --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-BB-LEVEL-ALL

; Test basic block level for one function
; RUN: llvm-ir2vec embeddings --level=bb --function=_Z3addii --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-BB-ADD-INT-INT

; Test instruction level for one function
; RUN: llvm-ir2vec embeddings --level=inst --function=_Z3addiii --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-INST-ADD-INT-INT-INT

; Test error case - non-existent function
; RUN: not llvm-ir2vec embeddings --level=func --function=_Z3subii --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s 2>&1 | FileCheck %s -check-prefix=CHECK-NONEXISTENT

; Function Attrs: mustprogress noinline nounwind optnone uwtable
; add(int, int)
define dso_local noundef i32 @_Z3addii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  store i32 %0, ptr %3, align 4
  store i32 %1, ptr %4, align 4
  %5 = load i32, ptr %3, align 4
  %6 = load i32, ptr %4, align 4
  %7 = add nsw i32 %5, %6
  ret i32 %7
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
; add(int, int, int)
define dso_local noundef i32 @_Z3addiii(i32 noundef %0, i32 noundef %1, i32 noundef %2) #0 {
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  store i32 %0, ptr %4, align 4
  store i32 %1, ptr %5, align 4
  store i32 %2, ptr %6, align 4
  %7 = load i32, ptr %4, align 4
  %8 = load i32, ptr %5, align 4
  %9 = add nsw i32 %7, %8
  %10 = load i32, ptr %6, align 4
  %11 = add nsw i32 %9, %10
  ret i32 %11
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
; add(double, double)
define dso_local noundef double @_Z3adddd(double noundef %0, double noundef %1) #0 {
  %3 = alloca double, align 8
  %4 = alloca double, align 8
  store double %0, ptr %3, align 8
  store double %1, ptr %4, align 8
  %5 = load double, ptr %3, align 8
  %6 = load double, ptr %4, align 8
  %7 = fadd double %5, %6
  ret double %7
}

; Function Attrs: mustprogress noinline norecurse nounwind optnone uwtable
define dso_local noundef i32 @main() #1 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca double, align 8
  store i32 0, ptr %1, align 4
  %5 = call noundef i32 @_Z3addii(i32 noundef 5, i32 noundef 3)
  store i32 %5, ptr %2, align 4
  %6 = call noundef i32 @_Z3addiii(i32 noundef 5, i32 noundef 3, i32 noundef 2)
  store i32 %6, ptr %3, align 4
  %7 = call noundef double @_Z3adddd(double noundef 5.500000e+00, double noundef 3.200000e+00)
  store double %7, ptr %4, align 8
  ret i32 0
}

; CHECK-DEFAULT: Function: _Z3addii
; CHECK-DEFAULT-NEXT: [ 602.00  610.00  618.00 ]
; CHECK-DEFAULT-NEXT: Function: _Z3addiii
; CHECK-DEFAULT-NEXT: [ 921.00  933.00  945.00 ]
; CHECK-DEFAULT-NEXT: Function: _Z3adddd
; CHECK-DEFAULT-NEXT: [ 605.00  613.00  621.00 ]
; CHECK-DEFAULT-NEXT: Function: main
; CHECK-DEFAULT-NEXT: [ 1251.00  1263.00  1275.00 ]

; CHECK-FUNC-LEVEL: Function: _Z3addii
; CHECK-FUNC-LEVEL-NEXT: [ 602.00  610.00  618.00 ]
; CHECK-FUNC-LEVEL-NEXT: Function: _Z3addiii
; CHECK-FUNC-LEVEL-NEXT: [ 921.00  933.00  945.00 ]
; CHECK-FUNC-LEVEL-NEXT: Function: _Z3adddd
; CHECK-FUNC-LEVEL-NEXT: [ 605.00  613.00  621.00 ]
; CHECK-FUNC-LEVEL-NEXT: Function: main
; CHECK-FUNC-LEVEL-NEXT: [ 1251.00  1263.00  1275.00 ]

; CHECK-ADD-INT-INT: Function: _Z3addii
; CHECK-ADD-INT-INT-NEXT: [ 602.00  610.00  618.00 ]

; CHECK-ADD-INT-INT-INT: Function: _Z3addiii
; CHECK-ADD-INT-INT-INT-NEXT: [ 921.00  933.00  945.00 ]

; CHECK-ADD-DOUBLE-DOUBLE: Function: _Z3adddd
; CHECK-ADD-DOUBLE-DOUBLE-NEXT: [ 605.00  613.00  621.00 ]

; CHECK-MAIN: Function: main
; CHECK-MAIN-NEXT: [ 1251.00  1263.00  1275.00 ]

; CHECK-DEMANGLED-INT-INT: Function: _Z3addii
; CHECK-DEMANGLED-INT-INT-NEXT: [ 602.00  610.00  618.00 ]

; CHECK-DEMANGLED-INT-INT-INT: Function: _Z3addiii
; CHECK-DEMANGLED-INT-INT-INT-NEXT: [ 921.00  933.00  945.00 ]

; CHECK-DEMANGLED-DOUBLE-DOUBLE: Function: _Z3adddd
; CHECK-DEMANGLED-DOUBLE-DOUBLE-NEXT: [ 605.00  613.00  621.00 ]

; CHECK-BB-LEVEL-ALL: Function: _Z3addii
; CHECK-BB-LEVEL-ALL-NEXT: [ 602.00  610.00  618.00 ]
; CHECK-BB-LEVEL-ALL-NEXT: Function: _Z3addiii
; CHECK-BB-LEVEL-ALL-NEXT: [ 921.00  933.00  945.00 ]
; CHECK-BB-LEVEL-ALL-NEXT: Function: _Z3adddd
; CHECK-BB-LEVEL-ALL-NEXT: [ 605.00  613.00  621.00 ]
; CHECK-BB-LEVEL-ALL-NEXT: Function: main
; CHECK-BB-LEVEL-ALL-NEXT: [ 1251.00  1263.00  1275.00 ]

; CHECK-BB-ADD-INT-INT: Function: _Z3addii
; CHECK-BB-ADD-INT-INT-NEXT: [ 602.00  610.00  618.00 ]

; CHECK-INST-ADD-INT-INT-INT: Function: _Z3addiii
; CHECK-INST-ADD-INT-INT-INT-NEXT: %4 = alloca i32, align 4 [ 91.00  92.00  93.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %5 = alloca i32, align 4 [ 91.00  92.00  93.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %6 = alloca i32, align 4 [ 91.00  92.00  93.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: store i32 %0, ptr %4, align 4 [ 97.00  98.00  99.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: store i32 %1, ptr %5, align 4 [ 97.00  98.00  99.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: store i32 %2, ptr %6, align 4 [ 97.00  98.00  99.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %7 = load i32, ptr %4, align 4 [ 94.00  95.00  96.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %8 = load i32, ptr %5, align 4 [ 94.00  95.00  96.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %9 = add nsw i32 %7, %8 [ 37.00  38.00  39.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %10 = load i32, ptr %6, align 4 [ 94.00  95.00  96.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: %11 = add nsw i32 %9, %10 [ 37.00  38.00  39.00 ]
; CHECK-INST-ADD-INT-INT-INT-NEXT: ret i32 %11 [ 1.00  2.00  3.00 ]

; CHECK-NONEXISTENT: error: Function '_Z3subii' not found