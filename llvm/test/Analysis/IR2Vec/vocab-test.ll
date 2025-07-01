; RUN: opt -passes='print<ir2vec-vocab>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_2D_vocab.json %s 2>&1 | FileCheck %s -check-prefix=VOCAB-CHECK
; RUN: opt -passes='print<ir2vec-vocab>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_2D_vocab.json -ir2vec-opc-weight=0.5 -ir2vec-type-weight=0.5 -ir2vec-arg-weight=0.5 %s 2>&1 | FileCheck %s -check-prefix=WT1-VOCAB-CHECK
; RUN: opt -passes='print<ir2vec-vocab>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_2D_vocab.json -ir2vec-opc-weight=0.1 -ir2vec-type-weight=0 -ir2vec-arg-weight=0 %s 2>&1 | FileCheck %s -check-prefix=WT2-VOCAB-CHECK
 
define dso_local void @test() {
  entry:
    ret void
}

; VOCAB-CHECK: Key: dummyArg: [ 0.20 0.40 ]
; VOCAB-CHECK-NEXT: Key: dummyOpc: [ 1.00 2.00 ]
; VOCAB-CHECK-NEXT: Key: dummyTy: [ 0.50 1.00 ]

; WT1-VOCAB-CHECK: Key: dummyArg: [ 0.50 1.00 ]
; WT1-VOCAB-CHECK-NEXT: Key: dummyOpc: [ 0.50 1.00 ]
; WT1-VOCAB-CHECK-NEXT: Key: dummyTy: [ 0.50 1.00 ]

; WT2-VOCAB-CHECK: Key: dummyArg: [ 0.00 0.00 ]
; WT2-VOCAB-CHECK-NEXT: Key: dummyOpc: [ 0.10 0.20 ]
; WT2-VOCAB-CHECK-NEXT: Key: dummyTy: [ 0.00 0.00 ]
