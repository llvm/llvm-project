; REQUIRES: asserts
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 \
; RUN:   -enable-aa-sched-mi -debug-only=machine-scheduler -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=AA
; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=x86-64-v3 \
; RUN:   -debug-only=machine-scheduler -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOAA

; With AA enabled, disjoint constant-offset accesses to the same identified
; object are indexed by byte range and do not receive MayAliasMem edges, while
; partially overlapping accesses ([4, 12) vs [0, 8) and [8, 16)) still do.
define i64 @disjoint_and_partial_overlap_aa(ptr noalias %p, i64 %a, i64 %b, i64 %c) {
; AA-LABEL: disjoint_and_partial_overlap_aa:%bb.0
; AA:       SU([[S0:[0-9]+]]): MOV64mr {{.*}} :: (store (s64) into %ir.p)
; AA:         Successors:
; AA-NEXT:      SU([[SMID:[0-9]+]]): Ord Latency=0 Memory
; AA-NOT:       Ord Latency={{[0-9]+}} Memory
; AA:       SU([[SMID]]): MOV64mr {{.*}} :: (store (s64) into %ir.gep4, align 4)
; AA:         Predecessors:
; AA-NEXT:      SU([[S0]]): Ord Latency=0 Memory
; AA:         Successors:
; AA-NEXT:      SU([[S1:[0-9]+]]): Ord Latency=0 Memory
; AA-NOT:       Ord Latency={{[0-9]+}} Memory
; AA:       SU([[S1]]): MOV64mr {{.*}} :: (store (s64) into %ir.gep8)
; AA:         Predecessors:
; AA-NEXT:      SU([[SMID]]): Ord Latency=0 Memory
; AA-NOT:       Ord Latency={{[0-9]+}} Memory
; AA:       SU([[L3:[0-9]+]]): {{.*}} = MOV64rm {{.*}} :: (load (s64) from %ir.gep24)
; AA-NOT:     Ord Latency={{[0-9]+}} Memory
; AA:         Successors:
  %gep4 = getelementptr inbounds i8, ptr %p, i64 4
  %gep8 = getelementptr inbounds i8, ptr %p, i64 8
  %gep24 = getelementptr inbounds i8, ptr %p, i64 24
  store i64 %a, ptr %p, align 8
  store i64 %b, ptr %gep4, align 4
  store i64 %c, ptr %gep8, align 8
  %v = load i64, ptr %gep24, align 8
  ret i64 %v
}

; Without AA, distinct-GEP disjoint stores to the same base object and base
; register prune transitively redundant MayAliasMem edges so each store links
; to at most the two most recent stores rather than all O(N) later stores.
define void @noaa_disjoint_reg_stores_pruned(ptr noalias %p, i64 %v0, i64 %v1, i64 %v2, i64 %v3, i64 %v4) {
; NOAA-LABEL: noaa_disjoint_reg_stores_pruned:%bb.0
; NOAA:       SU([[S0:[0-9]+]]): MOV64mr {{.*}} :: (store (s64) into %ir.p)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[S2:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NEXT:      SU([[S1:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[S1]]): MOV64mr {{.*}} :: (store (s64) into %ir.p1)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[S3:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NEXT:      SU([[S2]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[S2]]): MOV64mr {{.*}} :: (store (s64) into %ir.p2)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[S4:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NEXT:      SU([[S3]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[S3]]): MOV64mr {{.*}} :: (store (s64) into %ir.p3)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[S4]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[S4]]): MOV64mr {{.*}} :: (store (s64) into %ir.p4)
  %p1 = getelementptr inbounds i64, ptr %p, i64 1
  %p2 = getelementptr inbounds i64, ptr %p, i64 2
  %p3 = getelementptr inbounds i64, ptr %p, i64 3
  %p4 = getelementptr inbounds i64, ptr %p, i64 4
  store i64 %v0, ptr %p, align 8
  store i64 %v1, ptr %p1, align 8
  store i64 %v2, ptr %p2, align 8
  store i64 %v3, ptr %p3, align 8
  store i64 %v4, ptr %p4, align 8
  ret void
}

declare void @use_ptr(ptr)

define void @noaa_disjoint_frame_stores_pruned(i1 %cond, i64 %v0, i64 %v1, i64 %v2, i64 %v3, i64 %v4) {
; NOAA-LABEL: noaa_disjoint_frame_stores_pruned:%bb.0
; NOAA:       SU([[F0:[0-9]+]]): MOV64mr %stack.0.buf, {{.*}} :: (store (s64) into %ir.buf)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[F2:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NEXT:      SU([[F1:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[F1]]): MOV64mr %stack.0.buf, {{.*}} :: (store (s64) into %ir.p1)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[F3:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NEXT:      SU([[F2]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[F2]]): MOV64mr %stack.0.buf, {{.*}} :: (store (s64) into %ir.p2)
; NOAA:         Successors:
; NOAA-NEXT:      SU([[F4:[0-9]+]]): Ord Latency=0 Memory
; NOAA-NEXT:      SU([[F3]]): Ord Latency=0 Memory
; NOAA-NOT:       Ord Latency=0 Memory
; NOAA:       SU([[F3]]): MOV64mr %stack.0.buf, {{.*}} :: (store (s64) into %ir.p3)
entry:
  %buf = alloca [5 x i64], align 8
  %p1 = getelementptr inbounds i64, ptr %buf, i64 1
  %p2 = getelementptr inbounds i64, ptr %buf, i64 2
  %p3 = getelementptr inbounds i64, ptr %buf, i64 3
  %p4 = getelementptr inbounds i64, ptr %buf, i64 4
  store i64 %v0, ptr %buf, align 8
  store i64 %v1, ptr %p1, align 8
  store i64 %v2, ptr %p2, align 8
  store i64 %v3, ptr %p3, align 8
  store i64 %v4, ptr %p4, align 8
  br i1 %cond, label %callbb, label %exit

callbb:
  call void @use_ptr(ptr nonnull %buf)
  br label %exit

exit:
  ret void
}
