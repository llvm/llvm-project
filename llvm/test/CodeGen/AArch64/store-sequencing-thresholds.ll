; REQUIRES: asserts
; RUN: llc < %s -mtriple=arm64-apple-ios -debug-only=machine-scheduler \
; RUN:   -store-sequencing-max-frontier-size=3 \
; RUN:   -enable-store-sequencing 2>&1 | FileCheck %s --check-prefixes=CHECK,FRONTIER

; RUN: llc < %s -mtriple=arm64-apple-ios -debug-only=machine-scheduler \
; RUN:   -store-sequencing-max-memops-in-flight=5 \
; RUN:   -enable-store-sequencing 2>&1 | FileCheck %s --check-prefixes=CHECK,INFLIGHT

; RUN: llc < %s -mtriple=arm64-apple-ios -debug-only=machine-scheduler \
; RUN:   -store-sequencing-max-frontier-size=3 \
; RUN:   -store-sequencing-max-memops-in-flight=5 \
; RUN:   -enable-store-sequencing 2>&1 | FileCheck %s --check-prefixes=CHECK,BOTH

define void @test_oversize_unanalyzable_frontier_results_in_promotion(ptr %p) {
; CHECK-LABEL: test_oversize_unanalyzable_frontier_results_in_promotion:%bb.0

; FRONTIER:       Promoting SU([[S0:[0-9]+]]) to sequencing store
; FRONTIER:       Promoting SU([[S1:[0-9]+]]) to sequencing store
; FRONTIER:       SU([[S1]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.10)
; FRONTIER-NOT:   {{^}}SU(
; FRONTIER:       Successors:
; FRONTIER-NEXT:    SU([[S0]]): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER:       SU([[S0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.50)

; INFLIGHT:       Promoting SU([[S:[0-9]+]]) to sequencing store
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT:       SU([[B]]): STRWui {{.*}} :: (store (s32) into %ir.p)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[S]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.50)
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store i32 4, ptr %p.gep.40
  store i32 5, ptr %p.gep.50
  ret void
}

define i32 @test_noalias_unanalyzable_load_not_retained_across_promotion(ptr %p) {
; CHECK-LABEL: test_noalias_unanalyzable_load_not_retained_across_promotion:%bb.0

; FRONTIER:       Promoting SU([[S0:[0-9]+]]) to sequencing store
; FRONTIER:       Promoting SU([[S1:[0-9]+]]) to sequencing store
; FRONTIER:       SU([[S1]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.20)
; FRONTIER-NOT:   {{^}}SU(
; FRONTIER:       Successors:
; FRONTIER-NEXT:    SU([[S0]]): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Barrier
; FRONTIER:       SU([[L]]): {{.*}} = LDRWui {{.*}} :: (load (s32) from %ir.p.gep.60)
; FRONTIER:       SU([[S0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.50)

; INFLIGHT:       Promoting SU([[S0:[0-9]+]]) to sequencing store
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT:       Promoting SU([[S1:[0-9]+]]) to sequencing store
; INFLIGHT:       SU([[S1]]): STRWui {{.*}} :: (store (s32) into %ir.p)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU([[B]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[B]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.10)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Barrier
; INFLIGHT-NEXT:    SU([[S0]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[L]]): {{.*}} = LDRWui {{.*}} :: (load (s32) from %ir.p.gep.60)
; INFLIGHT:       SU([[S0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.50)
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50
  %p.gep.60 = getelementptr i32, ptr %p, i64 60

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store i32 4, ptr %p.gep.40
  %x = load i32, ptr %p.gep.60
  store i32 5, ptr %p.gep.50

  ret i32 %x
}

define void @test_oversize_analyzable_frontier_results_in_promotion(ptr noalias %p) {
; CHECK-LABEL: test_oversize_analyzable_frontier_results_in_promotion:%bb.0

; FRONTIER:       Promoting SU([[S:[0-9]+]]) to sequencing store
; FRONTIER:       SU([[S]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.20)
; FRONTIER-NOT:   {{^}}SU(
; FRONTIER:       Successors:
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; FRONTIER-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier

; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT:       SU([[B]]): STRWui {{.*}} :: (store (s32) into %ir.p)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store i32 4, ptr %p.gep.40
  store i32 5, ptr %p.gep.50
  ret void
}

define void @test_excess_memops_in_flight_introduce_barrier(ptr noalias %p, ptr noalias %q, ptr noalias %r, ptr noalias %s, ptr noalias %t, ptr %u) {
; CHECK-LABEL: test_excess_memops_in_flight_introduce_barrier:%bb.0
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT:       SU([[B]]): STRWui {{.*}} :: (store (s32) into %ir.u)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
  store i32 0, ptr %u ; unanalyzable
  store i32 1, ptr %p
  store i32 2, ptr %q
  store i32 3, ptr %r
  store i32 4, ptr %s
  store i32 5, ptr %t
  ret void
}

define void @test_excess_memops_in_flight_introduce_barrier_repeatedly(ptr noalias %p) {
; CHECK-LABEL: test_excess_memops_in_flight_introduce_barrier_repeatedly:%bb.0
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B0:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B1:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       SU([[B1]]): STRWui {{.*}} :: (store (s32) into %ir.p)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU([[B0]]): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT:       SU([[B0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.60)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50
  %p.gep.60 = getelementptr i32, ptr %p, i64 60
  %p.gep.70 = getelementptr i32, ptr %p, i64 70
  %p.gep.80 = getelementptr i32, ptr %p, i64 80
  %p.gep.90 = getelementptr i32, ptr %p, i64 90
  %p.gep.100 = getelementptr i32, ptr %p, i64 100
  %p.gep.110 = getelementptr i32, ptr %p, i64 110

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store i32 4, ptr %p.gep.40
  store i32 5, ptr %p.gep.50
  store i32 6, ptr %p.gep.60
  store i32 7, ptr %p.gep.70
  store i32 8, ptr %p.gep.80
  store i32 9, ptr %p.gep.90
  store i32 10, ptr %p.gep.100
  store i32 11, ptr %p.gep.110
  ret void
}

define void @test_promotion_discounts_memops_in_flight(ptr %p) {
; CHECK-LABEL: test_promotion_discounts_memops_in_flight:%bb.0

; INFLIGHT:       Promoting SU([[S0:[0-9]+]]) to sequencing store
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT:       Promoting SU([[S1:[0-9]+]]) to sequencing store
; INFLIGHT:       SU([[S1]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.20)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU([[B]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[B]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.30)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU([[S0]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[S0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.80)

; BOTH-NOT:   Number of in-flight
; BOTH:       Promoting SU([[S0:[0-9]+]]) to sequencing store
; BOTH-NOT:   Number of in-flight
; BOTH:       Promoting SU([[S1:[0-9]+]]) to sequencing store
; BOTH-NOT:   Number of in-flight
; BOTH:       Promoting SU([[S2:[0-9]+]]) to sequencing store
; BOTH-NOT:   {{Promoting|Number of in-flight}}
; BOTH:       SU([[S2]]): STRWui {{.*}} :: (store (s32) into %ir.p)
; BOTH-NOT:   {{^}}SU(
; BOTH:       Successors:
; BOTH-NEXT:    SU([[S1]]): Ord Latency=0 Barrier
; BOTH-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; BOTH-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; BOTH-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; BOTH:       SU([[S1]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.40)
; BOTH-NOT:   {{^}}SU(
; BOTH:       Successors:
; BOTH-NEXT:    SU([[S0]]): Ord Latency=0 Barrier
; BOTH-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; BOTH-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; BOTH-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; BOTH:       SU([[S0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.80)
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50
  %p.gep.60 = getelementptr i32, ptr %p, i64 60
  %p.gep.70 = getelementptr i32, ptr %p, i64 70
  %p.gep.80 = getelementptr i32, ptr %p, i64 80

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store i32 4, ptr %p.gep.40
  store i32 5, ptr %p.gep.50
  store i32 6, ptr %p.gep.60
  store i32 7, ptr %p.gep.70
  store i32 8, ptr %p.gep.80
  ret void
}

define i32 @test_retained_loads_count_towards_memops_in_flight(ptr %p, i64 %i) {
; CHECK-LABEL: test_retained_loads_count_towards_memops_in_flight:%bb.0
; INFLIGHT:       Promoting SU([[S:[0-9]+]]) to sequencing store
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       Number of in-flight memory operations reached 5, promoting SU([[B:[0-9]+]]) to barrier chain and clearing maps.
; INFLIGHT:       SU([[B]]): STRWui {{.*}} :: (store (s32) into %ir.p)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=0 Barrier
; INFLIGHT-NEXT:    SU([[L:[0-9]+]]): Ord Latency=1 Barrier
; INFLIGHT-NEXT:    SU([[S]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[S]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.40)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU({{[0-9]+}}): Ord Latency=1 Memory
; INFLIGHT:       SU([[L]]): {{.*}} = LDRWui {{.*}} :: (load (s32) from %ir.p.gep.50)
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50
  %p.gep.i = getelementptr i32, ptr %p, i64 %i

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store i32 4, ptr %p.gep.40
  %x = load i32, ptr %p.gep.50
  %y = load i32, ptr %p.gep.i
  %z = add i32 %x, %y
  ret i32 %z
}

define void @test_memops_in_flight_reset_after_barrier(ptr %p) {
; CHECK-LABEL: test_memops_in_flight_reset_after_barrier:%bb.0
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       Promoting SU([[S0:[0-9]+]]) to sequencing store
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       Global memory object and new barrier chain: SU([[V:[0-9]+]]).
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       Promoting SU([[S1:[0-9]+]]) to sequencing store
; INFLIGHT-NOT:   Number of in-flight
; INFLIGHT:       SU([[S1]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.30)
; INFLIGHT-NOT:   {{^}}SU(
; INFLIGHT:       Successors:
; INFLIGHT-NEXT:    SU([[V]]): Ord Latency=0 Barrier
; INFLIGHT:       SU([[V]]): STRWui {{.*}} :: (volatile store (s32) into %ir.p.gep.40)
; INFLIGHT:       SU([[S0]]): STRWui {{.*}} :: (store (s32) into %ir.p.gep.80)
  %p.gep.10 = getelementptr i32, ptr %p, i64 10
  %p.gep.20 = getelementptr i32, ptr %p, i64 20
  %p.gep.30 = getelementptr i32, ptr %p, i64 30
  %p.gep.40 = getelementptr i32, ptr %p, i64 40
  %p.gep.50 = getelementptr i32, ptr %p, i64 50
  %p.gep.60 = getelementptr i32, ptr %p, i64 60
  %p.gep.70 = getelementptr i32, ptr %p, i64 70
  %p.gep.80 = getelementptr i32, ptr %p, i64 80

  store i32 0, ptr %p
  store i32 1, ptr %p.gep.10
  store i32 2, ptr %p.gep.20
  store i32 3, ptr %p.gep.30
  store volatile i32 4, ptr %p.gep.40
  store i32 5, ptr %p.gep.50
  store i32 6, ptr %p.gep.60
  store i32 7, ptr %p.gep.70
  store i32 8, ptr %p.gep.80
  ret void
}
