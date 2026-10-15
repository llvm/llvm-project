;; Counter increments under TSan should be atomic to prevent it reporting
;; them as data races. The !nosanitize metadata should also be added to
;; prevent unnecessary (expensive) calls into the runtime.

; RUN: opt < %s -passes=instrprof -S | FileCheck %s
; RUN: opt < %s -passes=instrprof -do-counter-promotion -S | FileCheck %s --check-prefix=PROMO

target triple = "aarch64-unknown-linux-gnu"

@__profn_straight_line_tsan = private constant [18 x i8] c"straight_line_tsan"
@__profn_straight_line_no_tsan = private constant [21 x i8] c"straight_line_no_tsan"
@__profn_loop_tsan = private constant [9 x i8] c"loop_tsan"
@__profn_loop_no_tsan = private constant [12 x i8] c"loop_no_tsan"

define void @straight_line_tsan() sanitize_thread {
; CHECK-LABEL: define void @straight_line_tsan(
; CHECK-NEXT:    atomicrmw add ptr @__profc_straight_line_tsan, i64 1 monotonic, align 8, !nosanitize
; CHECK-NEXT:    ret void
;
  call void @llvm.instrprof.increment(ptr @__profn_straight_line_tsan, i64 0, i32 1, i32 0)
  ret void
}

define void @straight_line_no_tsan() {
; CHECK-LABEL: define void @straight_line_no_tsan() {
; CHECK-NEXT:    %[[LOAD:.+]] = load i64, ptr @__profc_straight_line_no_tsan, align 8, !nosanitize
; CHECK-NEXT:    %[[ADD:.+]] = add i64 %[[LOAD]], 1
; CHECK-NEXT:    store i64 %[[ADD]], ptr @__profc_straight_line_no_tsan, align 8, !nosanitize
; CHECK-NEXT:    ret void
;
  call void @llvm.instrprof.increment(ptr @__profn_straight_line_no_tsan, i64 0, i32 1, i32 0)
  ret void
}

;; With counter promotion, the loop counter is kept in a register and written
;; back once at the loop exit. Under TSan that write-back must also be atomic.

define void @loop_tsan(i32 %n) sanitize_thread {
; PROMO-LABEL: define void @loop_tsan(
; PROMO:       loop:
; PROMO-NEXT:    %[[PHI:.+]] = phi i64 [ 0, %preheader ], [ %[[NEXT:.+]], %loop ]
; PROMO-NOT:     atomicrmw
; PROMO:         %[[NEXT]] = add i64 %[[PHI]], 1
; PROMO-NOT:     atomicrmw
; PROMO:       loop.exit:
; PROMO-NEXT:    atomicrmw add ptr @__profc_loop_tsan, i64 %[[NEXT]] monotonic, align 8, !nosanitize
;
entry:
  call void @llvm.instrprof.increment(ptr @__profn_loop_tsan, i64 0, i32 2, i32 1)
  %enter = icmp sgt i32 %n, 0
  br i1 %enter, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %i = phi i32 [ 0, %preheader ], [ %inc, %loop ]
  call void @llvm.instrprof.increment(ptr @__profn_loop_tsan, i64 0, i32 2, i32 0)
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %loop.exit

loop.exit:
  br label %exit

exit:
  ret void
}

define void @loop_no_tsan(i32 %n) {
; PROMO-LABEL: define void @loop_no_tsan(
; PROMO:       loop:
; PROMO-NEXT:    %[[PHI2:.+]] = phi i64 [ 0, %preheader ], [ %[[NEXT2:.+]], %loop ]
; PROMO:         %[[NEXT2]] = add i64 %[[PHI2]], 1
; PROMO:       loop.exit:
; PROMO-NEXT:    %[[PROMOTED:.+]] = load i64, ptr @__profc_loop_no_tsan, align 8, !nosanitize
; PROMO-NEXT:    %[[SUM:.+]] = add i64 %[[PROMOTED]], %[[NEXT2]]
; PROMO-NEXT:    store i64 %[[SUM]], ptr @__profc_loop_no_tsan, align 8, !nosanitize
;
entry:
  call void @llvm.instrprof.increment(ptr @__profn_loop_no_tsan, i64 0, i32 2, i32 1)
  %enter = icmp sgt i32 %n, 0
  br i1 %enter, label %preheader, label %exit

preheader:
  br label %loop

loop:
  %i = phi i32 [ 0, %preheader ], [ %inc, %loop ]
  call void @llvm.instrprof.increment(ptr @__profn_loop_no_tsan, i64 0, i32 2, i32 0)
  %inc = add i32 %i, 1
  %cmp = icmp slt i32 %inc, %n
  br i1 %cmp, label %loop, label %loop.exit

loop.exit:
  br label %exit

exit:
  ret void
}
