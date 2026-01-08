; RUN: opt -passes='print<ir2vec>' -ir2vec-kind=flow-aware -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_opc_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK-OPC
; RUN: opt -passes='print<ir2vec>' -ir2vec-kind=flow-aware -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_type_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK-TYPE
; RUN: opt -passes='print<ir2vec>' -ir2vec-kind=flow-aware -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_arg_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK-ARG

define dso_local noundef float @_Z3abcif(i32 noundef %a, float noundef %b) #0 {
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

; 3D-CHECK-OPC: IR2Vec embeddings for function _Z3abcif:
; 3D-CHECK-OPC-NEXT: Function vector: [ 3630.00 3672.00 3714.00 ]
; 3D-CHECK-OPC-NEXT: Basic block vectors:
; 3D-CHECK-OPC-NEXT: Basic block: entry:
; 3D-CHECK-OPC-NEXT:  [ 3630.00 3672.00 3714.00 ]
; 3D-CHECK-OPC-NEXT: Instruction vectors:
; 3D-CHECK-OPC-NEXT: Instruction:   %a.addr = alloca i32, align 4 [ 91.00  92.00  93.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %b.addr = alloca float, align 4 [ 91.00  92.00  93.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   store i32 %a, ptr %a.addr, align 4 [ 188.00 190.00 192.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   store float %b, ptr %b.addr, align 4 [ 188.00 190.00 192.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %0 = load i32, ptr %a.addr, align 4 [ 185.00 187.00 189.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %1 = load i32, ptr %a.addr, align 4 [ 185.00 187.00 189.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %mul = mul nsw i32 %0, %1 [ 419.00  424.00  429.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %conv = sitofp i32 %mul to float [ 549.00  555.00  561.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %2 = load float, ptr %b.addr, align 4 [ 185.00  187.00  189.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %add = fadd float %conv, %2 [ 774.00  783.00  792.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   ret float %add [ 775.00  785.00  795.00 ]

; 3D-CHECK-TYPE: IR2Vec embeddings for function _Z3abcif:
; 3D-CHECK-TYPE-NEXT: Function vector:  [ 355.50  376.50  397.50 ]
; 3D-CHECK-TYPE-NEXT: Basic block vectors:
; 3D-CHECK-TYPE-NEXT: Basic block: entry:
; 3D-CHECK-TYPE-NEXT:  [ 355.50  376.50  397.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction vectors:
; 3D-CHECK-TYPE-NEXT: Instruction:   %a.addr = alloca i32, align 4 [ 12.50  13.00  13.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %b.addr = alloca float, align 4 [ 12.50  13.00  13.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   store i32 %a, ptr %a.addr, align 4 [ 14.50  15.50  16.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   store float %b, ptr %b.addr, align 4 [ 14.50  15.50  16.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %0 = load i32, ptr %a.addr, align 4 [ 22.00  23.00  24.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %1 = load i32, ptr %a.addr, align 4 [ 22.00  23.00  24.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %mul = mul nsw i32 %0, %1 [ 53.50  56.00  58.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %conv = sitofp i32 %mul to float [ 54.00  57.00  60.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %2 = load float, ptr %b.addr, align 4 [ 13.00  14.00  15.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %add = fadd float %conv, %2 [ 67.50  72.00  76.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   ret float %add [ 69.50  74.50  79.50 ]

; 3D-CHECK-ARG: IR2Vec embeddings for function _Z3abcif:
; 3D-CHECK-ARG-NEXT: Function vector:  [ 27.80  31.60  35.40 ]
; 3D-CHECK-ARG-NEXT: Basic block vectors:
; 3D-CHECK-ARG-NEXT: Basic block: entry:
; 3D-CHECK-ARG-NEXT:  [ 27.80  31.60  35.40 ]
; 3D-CHECK-ARG-NEXT: Instruction vectors:
; 3D-CHECK-ARG-NEXT: Instruction:   %a.addr = alloca i32, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %b.addr = alloca float, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   store i32 %a, ptr %a.addr, align 4 [ 3.40  3.80  4.20 ]
; 3D-CHECK-ARG-NEXT: Instruction:   store float %b, ptr %b.addr, align 4 [ 3.40  3.80  4.20 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %0 = load i32, ptr %a.addr, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %1 = load i32, ptr %a.addr, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %mul = mul nsw i32 %0, %1 [ 2.80  3.20  3.60 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %conv = sitofp i32 %mul to float [ 2.80  3.20  3.60 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %2 = load float, ptr %b.addr, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %add = fadd float %conv, %2 [ 4.20  4.80  5.40 ]
; 3D-CHECK-ARG-NEXT: Instruction:   ret float %add [ 4.20  4.80  5.40 ]
