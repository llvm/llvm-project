; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-apple-ios -debug-only=machine-scheduler -enable-unanalyzable-store-sequencing 2>&1 | FileCheck %s

define void @test_first_store_promoted(ptr %p) {
; CHECK-LABEL: test_first_store_promoted:%bb.0
; CHECK:       Promoting SU([[S:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU([[S]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  store i32 0, ptr %p
  ret void
}

define void @test_store_to_escaping_base_object_promoted(ptr %p, ptr %q) {
; CHECK-LABEL: test_store_to_escaping_base_object_promoted:%bb.0
; CHECK:       Promoting SU([[Q:[0-9]+]]) to sequencing store
; CHECK:       Promoting SU([[P:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU([[P]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[Q]]): Ord Latency=0 Barrier
; CHECK:       SU([[Q]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
  store i32 0, ptr %p
  store i32 0, ptr %q
  ret void
}

define void @test_repeated_stores_to_same_base_object_unpromoted(ptr %p, i64 %i) {
; CHECK-LABEL: test_repeated_stores_to_same_base_object_unpromoted:%bb.0
; CHECK:       Promoting SU([[S:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRWroX $wzr, %{{[0-9]+}}:gpr64common, %{{[0-9]+}}:gpr64, 0, 1 :: (store (s32) into %ir.p.gep.i)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[T:[0-9]+]]): Ord Latency=0 Memory
; CHECK:       SU([[T]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 2 :: (store (s32) into %ir.p.gep.2)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK:       SU([[S]]): STRXui $xzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s64) into %ir.p, align 4)
  %p.gep.i = getelementptr i32, ptr %p, i64 %i
  %p.gep.1 = getelementptr i32, ptr %p, i64 1
  %p.gep.2 = getelementptr i32, ptr %p, i64 2

  store i32 0, ptr %p.gep.i
  store i32 0, ptr %p.gep.1
  store i32 0, ptr %p.gep.2
  store i32 0, ptr %p
  ret void
}

define void @test_disjoint_frontier_stores_unordered(ptr %p, i64 %i) {
; CHECK-LABEL: test_disjoint_frontier_stores_unordered:%bb.0
; CHECK:       Promoting SU([[S:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK-NOT:     Ord
; CHECK:       SU({{[0-9]+}}): STRWui $wzr, %{{[0-9]+}}:gpr64common, 3 :: (store (s32) into %ir.p.gep.3)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK:       SU([[S]]): STRWroX $wzr, %{{[0-9]+}}:gpr64common, %{{[0-9]+}}:gpr64, 0, 1 :: (store (s32) into %ir.p.gep.i)
  %p.gep.i = getelementptr i32, ptr %p, i64 %i
  %p.gep.3 = getelementptr i32, ptr %p, i64 3

  store i32 0, ptr %p
  store i32 0, ptr %p.gep.3
  store i32 0, ptr %p.gep.i
  ret void
}

define i32 @test_intervening_load_from_escaping_object_triggers_promotion(ptr %p, ptr %q) {
; CHECK-LABEL: test_intervening_load_from_escaping_object_triggers_promotion:%bb.0
; CHECK:       Promoting SU([[B:[0-9]+]]) to sequencing store
; CHECK:       Promoting SU([[A:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU([[A]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[B]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Memory
; CHECK:       SU([[L]]): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 0 :: (load (s32) from %ir.q)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[B]]): Ord Latency=0 Barrier
; CHECK:       SU([[B]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  store i32 0, ptr %p
  %q.load = load i32, ptr %q
  store i32 0, ptr %p

  ret i32 %q.load
}

define i32 @test_non_escaping_load_does_not_trigger_promotion(ptr %p, i64 %i) {
; CHECK-LABEL: test_non_escaping_load_does_not_trigger_promotion:%bb.0
; CHECK:       Promoting SU([[S:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRWroX $wzr, %{{[0-9]+}}:gpr64common, %{{[0-9]+}}:gpr64, 0, 1 :: (store (s32) into %ir.p.gep.i)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Memory
; CHECK:       SU([[L]]): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 1 :: (load (s32) from %ir.p.gep.1)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK:       SU([[S]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  %p.gep.i = getelementptr i32, ptr %p, i64 %i
  %p.gep.1 = getelementptr i32, ptr %p, i64 1

  store i32 0, ptr %p.gep.i
  %p.load = load i32, ptr %p.gep.1
  store i32 0, ptr %p

  ret i32 %p.load
}

define i32 @test_non_aliasing_loads_retained_across_promotion(ptr %p, ptr %q, i64 %i) {
; CHECK-LABEL: test_non_aliasing_loads_retained_across_promotion:%bb.0
; CHECK:       Promoting SU([[Q:[0-9]+]]) to sequencing store
; CHECK:       Promoting SU([[P:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU([[P]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 1 :: (store (s32) into %ir.p.gep.1)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[Q]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Memory
; CHECK-NOT:     Ord
; CHECK:       SU({{[0-9]+}}): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 0 :: (load (s32) from %ir.p)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[Q]]): Ord Latency=0 Barrier
; CHECK:       SU([[L]]): %{{[0-9]+}}:gpr32 = LDRWroX %{{[0-9]+}}:gpr64common, %{{[0-9]+}}:gpr64, 0, 1 :: (load (s32) from %ir.p.gep.i)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[Q]]): Ord Latency=0 Barrier
; CHECK:       SU([[Q]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
  %p.gep.1 = getelementptr i32, ptr %p, i64 1
  %p.gep.i = getelementptr i32, ptr %p, i64 %i

  store i32 0, ptr %p.gep.1
  %p.load = load i32, ptr %p
  %p.gep.i.load = load i32, ptr %p.gep.i
  store i32 0, ptr %q

  %sum = add i32 %p.load, %p.gep.i.load
  ret i32 %sum
}

define i32 @test_retained_load_sequenced_against_preceding_store(ptr %p, ptr %q, i64 %i) {
; CHECK-LABEL: test_retained_load_sequenced_against_preceding_store:%bb.0
; CHECK:       Promoting SU([[Q:[0-9]+]]) to sequencing store
; CHECK:       Promoting SU([[P:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRWroX $wzr, %{{[0-9]+}}:gpr64common, %{{[0-9]+}}:gpr64, 0, 1 :: (store (s32) into %ir.p.gep.i)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[P]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Memory
; CHECK:       SU([[P]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 2 :: (store (s32) into %ir.p.gep.2)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[Q]]): Ord Latency=0 Barrier
; CHECK-NOT:     Ord
; CHECK:       SU([[L]]): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 0 :: (load (s32) from %ir.p)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[Q]]): Ord Latency=0 Barrier
; CHECK:       SU([[Q]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
  %p.gep.i = getelementptr i32, ptr %p, i64 %i
  %p.gep.2 = getelementptr i32, ptr %p, i64 2

  store i32 0, ptr %p.gep.i
  store i32 0, ptr %p.gep.2
  %p.load = load i32, ptr %p
  store i32 0, ptr %q

  ret i32 %p.load
}

define i32 @test_cleared_stores_reachable_through_sequencing_store(ptr %p, ptr %q, i64 %i) {
; CHECK-LABEL: test_cleared_stores_reachable_through_sequencing_store:%bb.0
; CHECK:       Promoting SU([[A:[0-9]+]]) to sequencing store
; CHECK:       Promoting SU([[N:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 0 :: (load (s32) from %ir.p)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[N]]): Ord Latency=0 Barrier
; CHECK-NOT:     Ord
; CHECK:       SU([[N]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[A]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[S:[0-9]+]]): Ord Latency=0 Barrier
; CHECK:       SU([[S]]): STRWroX $wzr, %{{[0-9]+}}:gpr64common, %{{[0-9]+}}:gpr64, 0, 1 :: (store (s32) into %ir.p.gep.i)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[A]]): Ord Latency=0 Barrier
; CHECK:       SU([[A]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  %p.gep.i = getelementptr i32, ptr %p, i64 %i

  %p.load = load i32, ptr %p
  store i32 0, ptr %q
  store i32 0, ptr %p.gep.i
  store i32 0, ptr %p

  ret i32 %p.load
}

define void @test_known_no_alias_store_precedes_sequencing_store(ptr noalias %p, ptr %q) {
; CHECK-LABEL: test_known_no_alias_store_precedes_sequencing_store:%bb.0
; CHECK:       Promoting SU([[C:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRXui $xzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s64) into %ir.p, align 4)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[C]]): Ord Latency=0 Barrier
; CHECK:       SU([[C]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
  %q.gep.1 = getelementptr i32, ptr %p, i64 1

  store i32 0, ptr %p
  store i32 0, ptr %q.gep.1
  store i32 0, ptr %q
  ret void
}

define i32 @test_known_no_alias_load_precedes_sequencing_store(ptr noalias %p, ptr %q) {
; CHECK-LABEL: test_known_no_alias_load_precedes_sequencing_store:%bb.0
; CHECK:       Promoting SU([[C:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 0 :: (load (s32) from %ir.p)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[C]]): Ord Latency=0 Barrier
; CHECK:       SU([[C]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
  %p.load = load i32, ptr %p
  store i32 0, ptr %q

  ret i32 %p.load
}

define i32 @test_known_store_sequenced_against_frontier_load(ptr noalias %p, ptr %q, i1 %c) {
; CHECK-LABEL: test_known_store_sequenced_against_frontier_load:%bb.0
; CHECK:       Promoting SU([[C:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[C]]): Ord Latency=0 Barrier
; CHECK-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Memory
; CHECK:       SU([[L]]): %{{[0-9]+}}:gpr32 = LDRWui %{{[0-9]+}}:gpr64common, 0 :: (load (s32) from %ir.p.or.q)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU({{[0-9]+}}): Data Latency=2 Reg=%{{[0-9]+}}
; CHECK-NEXT:    SU([[C]]): Ord Latency=0 Barrier
; CHECK:       SU([[C]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.q)
  %p.or.q = select i1 %c, ptr %p, ptr %q

  store i32 0, ptr %p
  %p.or.q.load = load i32, ptr %p.or.q
  store i32 0, ptr %q

  ret i32 %p.or.q.load
}

define void @test_global_memory_object_clears_frontier(ptr %p) {
; CHECK-LABEL: test_global_memory_object_clears_frontier:%bb.0
; CHECK:       Promoting SU([[B:[0-9]+]]) to sequencing store
; CHECK:       Global memory object and new barrier chain: SU([[F:[0-9]+]]).
; CHECK:       Promoting SU([[A:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU([[A]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[F]]): Ord Latency=1 Barrier
; CHECK:       SU([[F]]): DMB 11
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[B]]): Ord Latency=0 Barrier
; CHECK:       SU([[B]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  store i32 0, ptr %p
  fence seq_cst
  store i32 0, ptr %p
  ret void
}

define void @test_volatile_store_clears_frontier(ptr %p, ptr %q) {
; CHECK-LABEL: test_volatile_store_clears_frontier:%bb.0
; CHECK:       Promoting SU([[B:[0-9]+]]) to sequencing store
; CHECK:       Global memory object and new barrier chain: SU([[V:[0-9]+]]).
; CHECK:       Promoting SU([[A:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU([[A]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[V]]): Ord Latency=0 Barrier
; CHECK:       SU([[V]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (volatile store (s32) into %ir.q)
; CHECK-NOT:   {{^}}SU(
; CHECK:       Successors:
; CHECK-NEXT:    SU([[B]]): Ord Latency=0 Barrier
; CHECK:       SU([[B]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  store i32 0, ptr %p
  store volatile i32 0, ptr %q
  store i32 0, ptr %p
  ret void
}

define void @test_unordered_atomic_store_unpromoted(ptr %p, ptr %q) {
; CHECK-LABEL: test_unordered_atomic_store_unpromoted:%bb.0
; CHECK:       Promoting SU([[S:[0-9]+]]) to sequencing store
; CHECK-NOT:   Promoting

; CHECK:       SU({{[0-9]+}}): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store unordered (s32) into %ir.q)
; CHECK:       Successors:
; CHECK-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; CHECK:       SU([[S]]): STRWui $wzr, %{{[0-9]+}}:gpr64common, 0 :: (store (s32) into %ir.p)
  store atomic i32 0, ptr %q unordered, align 4
  store i32 0, ptr %p
  ret void
}
