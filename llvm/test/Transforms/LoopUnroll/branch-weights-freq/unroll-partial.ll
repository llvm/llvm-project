; Test branch weight metadata, estimated trip count metadata, and block
; frequencies after partial loop unrolling without -unroll-runtime.

; RUN: opt < %s -S -passes='print<block-freq>' 2>&1 | \
; RUN:   FileCheck -check-prefix=CHECK %s

; The -implicit-check-not options make sure that no additional labels or calls
; to @f show up.
; RUN: opt < %s -S -passes='loop-unroll,print<block-freq>' \
; RUN:     -unroll-count=4 2>&1 | \
; RUN:   FileCheck %s -check-prefix=CHECK-UR \
; RUN:       -implicit-check-not='{{^( *- )?[^ ;]*:}}' \
; RUN:       -implicit-check-not='call void @f'

; CHECK: block-frequency-info: test
; CHECK: do.body: float = 10.0,

; The sum should still be ~10.
;
; CHECK-UR: block-frequency-info: test
; CHECK-UR: - [[ENTRY:.*]]:
; CHECK-UR: - [[DO_BODY:.*]]: float = 2.9078,
; CHECK-UR: - [[DO_BODY_1:.*]]: float = 2.617,
; CHECK-UR: - [[DO_BODY_2:.*]]: float = 2.3553,
; CHECK-UR: - [[DO_BODY_3:.*]]: float = 2.1198,
; CHECK-UR: - [[DO_END:.*]]:

declare void @f(i32)

define void @test(i32 %n) {
; CHECK-UR-LABEL: define void @test(i32 %{{.*}}) {
;       CHECK-UR: [[ENTRY]]:
;       CHECK-UR:   br label %[[DO_BODY]]
;       CHECK-UR: [[DO_BODY]]:
;       CHECK-UR:   call void @f
;       CHECK-UR:   br i1 %{{.*}}, label %[[DO_END]], label %[[DO_BODY_1]], !prof ![[#PROF:]]
;       CHECK-UR: [[DO_BODY_1]]:
;       CHECK-UR:   call void @f
;       CHECK-UR:   br i1 %{{.*}}, label %[[DO_END]], label %[[DO_BODY_2]], !prof ![[#PROF]]
;       CHECK-UR: [[DO_BODY_2]]:
;       CHECK-UR:   call void @f
;       CHECK-UR:   br i1 %{{.*}}, label %[[DO_END]], label %[[DO_BODY_3]], !prof ![[#PROF]]
;       CHECK-UR: [[DO_BODY_3]]:
;       CHECK-UR:   call void @f
;       CHECK-UR:   br i1 %{{.*}}, label %[[DO_END]], label %[[DO_BODY]], !prof ![[#PROF]], !llvm.loop ![[#LOOP_UR_LATCH:]]
;       CHECK-UR: [[DO_END]]:
;       CHECK-UR:   ret void

entry:
  br label %do.body

do.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %do.body ]
  %inc = add i32 %i, 1
  call void @f(i32 %i)
  %c = icmp sge i32 %inc, %n
  br i1 %c, label %do.end, label %do.body, !prof !0

do.end:
  ret void
}

!0 = !{!"branch_weights", i32 1, i32 9}

; CHECK-UR: ![[#PROF]] = !{!"branch_weights", i32 1, i32 9}
; CHECK-UR: ![[#LOOP_UR_LATCH]] = distinct !{![[#LOOP_UR_LATCH]], ![[#LOOP_UR_TC:]], ![[#DISABLE:]]}
; CHECK-UR: ![[#LOOP_UR_TC]] = !{!"llvm.loop.estimated_trip_count", i32 3}
; CHECK-UR: ![[#DISABLE]] = !{!"llvm.loop.unroll.disable"}
