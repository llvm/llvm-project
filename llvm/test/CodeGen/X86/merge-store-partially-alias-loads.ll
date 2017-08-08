; REQUIRES: asserts
; RUN: llc -march=x86-64 -mtriple=x86_64-unknown-linux-gnu < %s | FileCheck -check-prefix=X86 %s
; RUN: llc -march=x86-64 -mtriple=x86_64-unknown-linux-gnu -debug-only=isel < %s 2>&1 | FileCheck -check-prefix=DBGDAG %s

; It's OK to merge the load / store of the first 2 components, but
; they must not be placed on the same chain after merging.

; X86-LABEL: {{^}}merge_store_partial_overlap_load:
; X86-DAG: movzwl ([[BASEREG:%[a-z]+]]), %e[[LO2:[a-z]+]]
; X86-DAG: movb 2([[BASEREG]]), [[HI1:%[a-z]+]]

; X86-NEXT: movw %[[LO2]], 1([[BASEREG]])
; X86-NEXT: movb [[HI1]], 3([[BASEREG]])
; X86-NEXT: retq

; DBGDAG-LABEL: Optimized lowered selection DAG: BB#0 'merge_store_partial_overlap_load:'
; DBGDAG: [[ENTRYTOKEN:t[0-9]+]]: ch = EntryToken
; DBGDAG-DAG: [[BASEPTR:t[0-9]+]]: i64,ch = CopyFromReg [[ENTRYTOKEN]],
; DBGDAG-DAG: [[ADDPTR:t[0-9]+]]: i64 = add [[BASEPTR]], Constant:i64<2>

; DBGDAG-DAG: [[LD2:t[0-9]+]]: i16,ch = load<LD2[%tmp81](align=1)> [[ENTRYTOKEN]], [[BASEPTR]], undef:i64
; DBGDAG-DAG: [[LD1:t[0-9]+]]: i8,ch = load<LD1[%tmp12]> [[ENTRYTOKEN]], [[ADDPTR]], undef:i64

; DBGDAG-DAG: [[ST1:t[0-9]+]]: ch = store<ST1[%tmp14]> [[ENTRYTOKEN]], [[LD1]], t{{[0-9]+}}, undef:i64
; DBGDAG-DAG: [[LOADTOKEN:t[0-9]+]]: ch = TokenFactor [[LD2]]:1, [[LD1]]:1
; DBGDAG-DAG: [[ST2:t[0-9]+]]: ch = store<ST2[%tmp10](align=1)> [[LOADTOKEN]], [[LD2]], t{{[0-9]+}}, undef:i64

; DBGDAG: X86ISD::RET_FLAG t{{[0-9]+}},

; DBGDAG: Type-legalized selection DAG: BB#0 'merge_store_partial_overlap_load:'
define void @merge_store_partial_overlap_load([4 x i8]* %tmp) {
  %tmp8 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i8 0
  %tmp10 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i8 1
  %tmp12 = getelementptr inbounds [4 x i8], [4 x i8]* %tmp, i32 0, i8 2
  %tmp14 = getelementptr [4 x i8], [4 x i8]* %tmp, i32 0, i8 3

  %tmp9 = load i8, i8* %tmp8, align 1   ; base + 0
  %tmp11 = load i8, i8* %tmp10, align 1 ; base + 1
  %tmp13 = load i8, i8* %tmp12, align 1 ; base + 2

  store i8 %tmp9, i8* %tmp10, align 1   ; base + 1
  store i8 %tmp11, i8* %tmp12, align 1  ; base + 2
  store i8 %tmp13, i8* %tmp14, align 1  ; base + 3

; Should emit
; load base + 0, base + 1
; store base + 1, base + 2
; load base + 2
; store base + 3

  ret void
}
