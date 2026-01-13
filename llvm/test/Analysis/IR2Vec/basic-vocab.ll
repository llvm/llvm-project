; RUN: not opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/incorrect_vocab1.json %s 2>&1 | FileCheck %s -check-prefix=INCORRECT-VOCAB1-CHECK
; RUN: not opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/incorrect_vocab2.json %s 2>&1 | FileCheck %s -check-prefix=INCORRECT-VOCAB2-CHECK
; RUN: not opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/incorrect_vocab3.json %s 2>&1 | FileCheck %s -check-prefix=INCORRECT-VOCAB3-CHECK
; RUN: not opt -passes='print<ir2vec>' -o /dev/null -ir2vec-vocab-path=%S/Inputs/incorrect_vocab4.json %s 2>&1 | FileCheck %s -check-prefix=INCORRECT-VOCAB4-CHECK
 
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

; INCORRECT-VOCAB1-CHECK: error: Error reading vocabulary: Missing 'Opcodes' section in vocabulary file

; INCORRECT-VOCAB2-CHECK: error: Error reading vocabulary: Missing 'Types' section in vocabulary file

; INCORRECT-VOCAB3-CHECK: error: Error reading vocabulary: Missing 'Arguments' section in vocabulary file

; INCORRECT-VOCAB4-CHECK: error: Error reading vocabulary: Vocabulary sections have different dimensions
