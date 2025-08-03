; RUN: llvm-ir2vec embeddings --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-DEFAULT
; RUN: llvm-ir2vec embeddings --level=func --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-FUNC-LEVEL
; RUN: llvm-ir2vec embeddings --level=func --function=abc --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-FUNC-LEVEL-ABC
; RUN: not llvm-ir2vec embeddings --level=func --function=def --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s 2>&1 | FileCheck %s -check-prefix=CHECK-FUNC-DEF
; RUN: llvm-ir2vec embeddings --level=bb --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-BB-LEVEL
; RUN: llvm-ir2vec embeddings --level=bb --function=abc_repeat --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-BB-LEVEL-ABC-REPEAT
; RUN: llvm-ir2vec embeddings --level=inst --function=abc_repeat --ir2vec-vocab-path=%ir2vec_test_vocab_dir/dummy_3D_nonzero_opc_vocab.json %s | FileCheck %s -check-prefix=CHECK-INST-LEVEL-ABC-REPEAT

define dso_local noundef float @abc(i32 noundef %a, float noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca float, align 4
  store i32 %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %a.addr, align 4
  %mul = mul nsw i32 %0, %1
  %conv = sitofp i32 %mul to float
  %2 = load float, ptr %b.addr, align 4
  %add = fadd float %conv, %2
  ret float %add
}

define dso_local noundef float @abc_repeat(i32 noundef %a, float noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca float, align 4
  store i32 %a, ptr %a.addr, align 4
  store float %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %a.addr, align 4
  %mul = mul nsw i32 %0, %1
  %conv = sitofp i32 %mul to float
  %2 = load float, ptr %b.addr, align 4
  %add = fadd float %conv, %2
  ret float %add
}

; CHECK-DEFAULT: Function: abc
; CHECK-DEFAULT-NEXT: [ 878.00  889.00  900.00 ]
; CHECK-DEFAULT-NEXT: Function: abc_repeat
; CHECK-DEFAULT-NEXT: [ 878.00  889.00  900.00 ]

; CHECK-FUNC-LEVEL: Function: abc 
; CHECK-FUNC-LEVEL-NEXT: [ 878.00  889.00  900.00 ]
; CHECK-FUNC-LEVEL-NEXT: Function: abc_repeat 
; CHECK-FUNC-LEVEL-NEXT: [ 878.00  889.00  900.00 ]

; CHECK-FUNC-LEVEL-ABC: Function: abc
; CHECK-FUNC-LEVEL-NEXT-ABC:  [ 878.00  889.00  900.00 ]

; CHECK-FUNC-DEF: Error: Function 'def' not found

; CHECK-BB-LEVEL: Function: abc
; CHECK-BB-LEVEL-NEXT: entry: [ 878.00  889.00  900.00 ]
; CHECK-BB-LEVEL-NEXT: Function: abc_repeat
; CHECK-BB-LEVEL-NEXT: entry: [ 878.00  889.00  900.00 ]

; CHECK-BB-LEVEL-ABC-REPEAT: Function: abc_repeat
; CHECK-BB-LEVEL-ABC-REPEAT-NEXT: entry: [ 878.00  889.00  900.00 ]

; CHECK-INST-LEVEL-ABC-REPEAT: Function: abc_repeat
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %a.addr = alloca i32, align 4 [ 91.00  92.00  93.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %b.addr = alloca float, align 4 [ 91.00  92.00  93.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: store i32 %a, ptr %a.addr, align 4 [ 97.00  98.00  99.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: store float %b, ptr %b.addr, align 4 [ 97.00  98.00  99.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %0 = load i32, ptr %a.addr, align 4 [ 94.00  95.00  96.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %1 = load i32, ptr %a.addr, align 4 [ 94.00  95.00  96.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %mul = mul nsw i32 %0, %1 [ 49.00  50.00  51.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %conv = sitofp i32 %mul to float [ 130.00  131.00  132.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %2 = load float, ptr %b.addr, align 4 [ 94.00  95.00  96.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: %add = fadd float %conv, %2 [ 40.00  41.00  42.00 ]
; CHECK-INST-LEVEL-ABC-REPEAT-NEXT: ret float %add [ 1.00  2.00  3.00 ]
