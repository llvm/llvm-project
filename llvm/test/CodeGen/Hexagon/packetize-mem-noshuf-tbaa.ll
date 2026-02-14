; RUN: llc -march=hexagon -mcpu=hexagonv65 -O2 < %s | FileCheck %s
; RUN: llc -march=hexagon -mcpu=hexagonv62 -O2 < %s | FileCheck %s --check-prefix=V62

; When the scheduler uses TBAA to conclude that a store and load access
; different types, it omits the Order (memory) dependency edge.  On V65+
; the packetizer can then place the store in slot 1 and the load in slot
; 0 of the same packet.  Without :mem_noshuf the hardware is free to
; reorder the memory operations, which is unsound when the pointers
; actually alias at runtime (TBAA can be overly optimistic with
; type-punning patterns such as libc++ tree-node pointer casts).
;
; These tests verify that the packetizer adds :mem_noshuf when it
; re-checks aliasing without TBAA and the accesses may alias.

;--- Tree node insertion pattern ---
;
; Models the inlined std::set::__insert_node_at sequence where a store
; to new_node->__parent_ (through one pointer type) is followed by a
; load from begin_node->__left_ (through a different pointer type).
; TBAA says the accesses are to different types, but the pointers may
; alias at runtime because libc++ tree nodes cast between base/derived
; node types.

; CHECK-LABEL: test_tree_node_insert:
; CHECK:       {
; CHECK-DAG:   memw(r0+#0) = r1
; CHECK-DAG:   r{{[0-9]+}} = memw(r2+#0)
; CHECK:       } :mem_noshuf

; V62-LABEL: test_tree_node_insert:
; V62-NOT:   :mem_noshuf

define ptr @test_tree_node_insert(ptr %new_node, ptr %parent, ptr %child_ptr) #0 {
entry:
  store ptr %parent, ptr %new_node, align 4, !tbaa !0
  %child = load ptr, ptr %child_ptr, align 4, !tbaa !3
  ret ptr %child
}

;--- Compound instruction logic ---
;
; Models a function where a store and load with different TBAA types
; are followed by a comparison and branch.  When the store-load pair
; is miscompiled (wrong memory ordering), the loaded value used for
; the comparison is stale, causing incorrect control-flow decisions.
; In the original bug this broke compound instruction splitting in
; HexagonMCCompound.cpp.

; CHECK-LABEL: test_store_load_branch:
; CHECK:       {
; CHECK-DAG:   memw(r0+#0) = r2
; CHECK-DAG:   r{{[0-9]+}} = memw(r1+#0)
; CHECK:       } :mem_noshuf

; V62-LABEL: test_store_load_branch:
; V62-NOT:   :mem_noshuf

define i32 @test_store_load_branch(ptr %flag_ptr, ptr %data_ptr, i32 %val) #0 {
entry:
  store i32 %val, ptr %flag_ptr, align 4, !tbaa !0
  %data = load i32, ptr %data_ptr, align 4, !tbaa !3
  %cmp = icmp eq i32 %data, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:
  %sum = add i32 %val, 1
  ret i32 %sum

if.end:
  ret i32 %data
}

;--- Simple store-load with immediate ---
;
; The simplest case: store an immediate, load through a different
; pointer.  This models the Verifier.cpp miscompile where even trivial
; IR triggered false "Broken module" errors because a stored boolean
; flag was read back through a differently-typed pointer with stale
; data.

; CHECK-LABEL: test_store_imm_load:
; CHECK:       {
; CHECK-DAG:   memw(r0+#0) = #1
; CHECK-DAG:   r0 = memw(r1+#0)
; CHECK:       } :mem_noshuf

; V62-LABEL: test_store_imm_load:
; V62-NOT:   :mem_noshuf

define i32 @test_store_imm_load(ptr %p, ptr %q) #0 {
entry:
  store i32 1, ptr %p, align 4, !tbaa !0
  %v = load i32, ptr %q, align 4, !tbaa !3
  ret i32 %v
}

;--- Anti-dependency path ---
;
; Models the LiveIntervals/MachineRegisterInfo pattern where an
; iterator's pointer is loaded (defining a register), then a store
; through a different TBAA type uses overlapping registers.  The
; scheduler creates an Anti dependency (register reuse) but no Order
; dependency (TBAA says different types), so the pair can land in the
; same packet.

; CHECK-LABEL: test_anti_dep_path:
; CHECK:       {
; CHECK-DAG:   r{{[0-9]+}} = memw(r1+#0)
; CHECK-DAG:   memw(r0+#0) = #42
; CHECK:       } :mem_noshuf

; V62-LABEL: test_anti_dep_path:
; V62-NOT:   :mem_noshuf

define i32 @test_anti_dep_path(ptr %p, ptr %q) #0 {
entry:
  %addr = load ptr, ptr %q, align 4, !tbaa !3
  store i32 42, ptr %p, align 4, !tbaa !0
  %v = load i32, ptr %addr, align 4, !tbaa !3
  ret i32 %v
}

;--- Multiple store-load pairs (systemic) ---
;
; Models the systemic miscompilation pattern where multiple unrelated
; store-load pairs in the same function all need :mem_noshuf
; protection.  Each pair uses a different TBAA type combination,
; modeling accesses to different C++ class hierarchies that share
; memory through pointer casts.

; CHECK-LABEL: test_multi_store_load:
; CHECK:       {
; CHECK-DAG:   memw(r0+#0) = #10
; CHECK-DAG:   r{{[0-9]+}} = memw(r1+#0)
; CHECK:       } :mem_noshuf
; CHECK:       {
; CHECK-DAG:   memw(r2+#0) = r
; CHECK-DAG:   r{{[0-9]+}} = memw(r3+#0)
; CHECK:       } :mem_noshuf

; V62-LABEL: test_multi_store_load:
; V62-NOT:   :mem_noshuf

define i32 @test_multi_store_load(ptr %p1, ptr %p2, ptr %p3, ptr %p4) #0 {
entry:
  store i32 10, ptr %p1, align 4, !tbaa !0
  %v1 = load i32, ptr %p2, align 4, !tbaa !3
  store i32 %v1, ptr %p3, align 4, !tbaa !5
  %v2 = load i32, ptr %p4, align 4, !tbaa !7
  %sum = add i32 %v1, %v2
  ret i32 %sum
}

attributes #0 = { nounwind }

; TBAA type hierarchy: four unrelated types under the same root.
; The scheduler sees these as non-aliasing, but they may alias at
; runtime through pointer casts (e.g., libc++ tree node base/derived).
!0 = !{!1, !1, i64 0}        ; type_a
!1 = !{!"type_a", !2}
!2 = !{!"tbaa_root"}
!3 = !{!4, !4, i64 0}        ; type_b
!4 = !{!"type_b", !2}
!5 = !{!6, !6, i64 0}        ; type_c
!6 = !{!"type_c", !2}
!7 = !{!8, !8, i64 0}        ; type_d
!8 = !{!"type_d", !2}
