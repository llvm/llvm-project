; REQUIRES: asserts
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -dag-maps-huge-region=2 2>&1 | FileCheck %s --check-prefix=THR2
; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu -debug-only=machine-scheduler -dag-maps-huge-region=3 2>&1 | FileCheck %s --check-prefix=THR3

define void @test_barrier(ptr %p) {
entry:
  %p10 = getelementptr i32, ptr %p, i64 10
  %p20 = getelementptr i32, ptr %p, i64 20
  %p30 = getelementptr i32, ptr %p, i64 30

  store i32 1, ptr %p
  store i32 2, ptr %p10
  store i32 3, ptr %p20
  store i32 4, ptr %p30
  ret void
}

; %p is a barrier. successors are %p10 and %p20 (previous barrier) due to being a barrier
; THR2:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p)
; THR2:      # succs left       : 2
; THR2:      Successors:
; THR2-DAG:   SU({{.*}}): Ord  Latency=0 Barrier
; THR2-DAG:   SU({{.*}}): Ord  Latency=0 Barrier

; %p10 is not a barrier. successors are %p20 (previous barrier)
; THR2:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p10)
; THR2:      # succs left       : 1
; THR2:      Successors:
; THR2-NEXT:   SU({{.*}}): Ord  Latency=0 Barrier

; %p20 is a barrier. successors are %p30 due to being a barrier
; THR2:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p20)
; THR2:      # succs left       : 1
; THR2:      Successors:
; THR2-NEXT:   SU({{.*}}): Ord  Latency=0 Barrier

; THR2:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p30)
; THR2:      # succs left       : 0


; %p is not a barrier. successors are %p10 (previous barrier)
; THR3:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p)
; THR3:      # succs left       : 1
; THR3:      Successors:
; THR3-NEXT:   SU({{.*}}): Ord  Latency=0 Barrier

; %p10 is a barrier. successors are %p20 and %p30 due to being a barrier
; THR3:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p10)
; THR3:      # succs left       : 2
; THR3:      Successors:
; THR3-DAG:   SU({{.*}}): Ord  Latency=0 Barrier
; THR3-DAG:   SU({{.*}}): Ord  Latency=0 Barrier

; %p20 is not a barrier. no successors
; THR3:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p20)
; THR3:      # succs left       : 0

; %p30 is not a barrier. no successors
; THR3:      SU({{.*}}):   STRWui {{.*}} :: (store (s32) into %ir.p30)
; THR3:      # succs left       : 0
