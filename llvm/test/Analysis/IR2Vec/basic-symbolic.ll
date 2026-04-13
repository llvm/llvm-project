; RUN: opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_opc_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK-OPC
; RUN: opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_type_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK-TYPE
; RUN: opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/dummy_3D_nonzero_arg_vocab.json %s 2>&1 | FileCheck %s -check-prefix=3D-CHECK-ARG

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
; 3D-CHECK-OPC-NEXT: Function vector: [ 878.00  889.00  900.00 ]
; 3D-CHECK-OPC-NEXT: Basic block vectors:
; 3D-CHECK-OPC-NEXT: Basic block: entry:
; 3D-CHECK-OPC-NEXT:  [ 878.00  889.00  900.00 ]
; 3D-CHECK-OPC-NEXT: Instruction vectors:
; 3D-CHECK-OPC-NEXT: Instruction:   %a.addr = alloca i32, align 4 [ 91.00  92.00  93.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %b.addr = alloca float, align 4 [ 91.00  92.00  93.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   store i32 %a, ptr %a.addr, align 4 [ 97.00  98.00  99.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   store float %b, ptr %b.addr, align 4 [ 97.00  98.00  99.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %0 = load i32, ptr %a.addr, align 4 [ 94.00  95.00  96.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %1 = load i32, ptr %a.addr, align 4 [ 94.00  95.00  96.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %mul = mul nsw i32 %0, %1 [ 49.00  50.00  51.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %conv = sitofp i32 %mul to float [ 130.00  131.00  132.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %2 = load float, ptr %b.addr, align 4 [ 94.00  95.00  96.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   %add = fadd float %conv, %2 [ 40.00  41.00  42.00 ]
; 3D-CHECK-OPC-NEXT: Instruction:   ret float %add [ 1.00  2.00  3.00 ]

; 3D-CHECK-TYPE: IR2Vec embeddings for function _Z3abcif:
; 3D-CHECK-TYPE-NEXT: Function vector: [ 61.00  66.50  72.00 ]
; 3D-CHECK-TYPE-NEXT: Basic block vectors:
; 3D-CHECK-TYPE-NEXT: Basic block: entry:
; 3D-CHECK-TYPE-NEXT:  [ 61.00  66.50  72.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction vectors:
; 3D-CHECK-TYPE-NEXT: Instruction:   %a.addr = alloca i32, align 4 [ 12.50  13.00  13.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %b.addr = alloca float, align 4 [ 12.50  13.00  13.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   store i32 %a, ptr %a.addr, align 4 [ 2.00  2.50  3.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   store float %b, ptr %b.addr, align 4 [ 2.00  2.50  3.00 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %0 = load i32, ptr %a.addr, align 4 [ 9.50  10.00  10.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %1 = load i32, ptr %a.addr, align 4 [ 9.50  10.00  10.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %mul = mul nsw i32 %0, %1 [ 9.50  10.00  10.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %conv = sitofp i32 %mul to float [ 0.50  1.00  1.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %2 = load float, ptr %b.addr, align 4 [ 0.50  1.00  1.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   %add = fadd float %conv, %2 [ 0.50  1.00  1.50 ]
; 3D-CHECK-TYPE-NEXT: Instruction:   ret float %add [ 2.00  2.50  3.00 ]

; 3D-CHECK-ARG: IR2Vec embeddings for function _Z3abcif:
; 3D-CHECK-ARG-NEXT: Function vector: [ 22.80  25.80  28.80 ]
; 3D-CHECK-ARG-NEXT: Basic block vectors:
; 3D-CHECK-ARG-NEXT: Basic block: entry:
; 3D-CHECK-ARG-NEXT:  [ 22.80  25.80  28.80 ]
; 3D-CHECK-ARG-NEXT: Instruction vectors:
; 3D-CHECK-ARG-NEXT: Instruction:   %a.addr = alloca i32, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %b.addr = alloca float, align 4 [ 1.40  1.60  1.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   store i32 %a, ptr %a.addr, align 4 [ 2.80  3.20  3.60 ]
; 3D-CHECK-ARG-NEXT: Instruction:   store float %b, ptr %b.addr, align 4 [ 2.80  3.20  3.60 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %0 = load i32, ptr %a.addr, align 4 [ 0.80  1.00  1.20 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %1 = load i32, ptr %a.addr, align 4 [ 0.80  1.00  1.20 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %mul = mul nsw i32 %0, %1 [ 4.00  4.40  4.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %conv = sitofp i32 %mul to float [ 2.00  2.20  2.40 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %2 = load float, ptr %b.addr, align 4 [ 0.80  1.00  1.20 ]
; 3D-CHECK-ARG-NEXT: Instruction:   %add = fadd float %conv, %2 [ 4.00  4.40  4.80 ]
; 3D-CHECK-ARG-NEXT: Instruction:   ret float %add [ 2.00  2.20  2.40 ]
