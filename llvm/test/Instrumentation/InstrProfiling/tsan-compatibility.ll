;; Counter increments under TSan should be atomic to prevent it reporting
;; them as data races. The !nosanitize metadata should also be added to
;; prevent unnecessary (expensive) calls into the runtime.

; RUN: opt < %s -passes=instrprof -S | FileCheck %s --check-prefixes=CHECK,NOCOND
; RUN: opt < %s -passes=instrprof -conditional-counter-update -S | FileCheck %s --check-prefixes=CHECK,COND
; RUN: opt < %s -passes=instrprof -do-counter-promotion -S | FileCheck %s --check-prefix=PROMO

target triple = "aarch64-unknown-linux-gnu"

@__profn_straight_line_tsan = private constant [18 x i8] c"straight_line_tsan"
@__profn_straight_line_no_tsan = private constant [21 x i8] c"straight_line_no_tsan"
@__profn_loop_tsan = private constant [9 x i8] c"loop_tsan"
@__profn_loop_no_tsan = private constant [12 x i8] c"loop_no_tsan"
@__profn_mcdc_tsan = private constant [9 x i8] c"mcdc_tsan"
@__profn_mcdc_no_tsan = private constant [12 x i8] c"mcdc_no_tsan"

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

;; Under TSan the coverage store and the MC/DC bitmap load and update should be
;; atomic. With -conditional-counter-update the coverage store is guarded by a
;; load of the counter, which must also be atomic.

define dso_local void @mcdc_tsan(i32 noundef %A) sanitize_thread {
; CHECK-LABEL: define dso_local void @mcdc_tsan(
; NOCOND:        store atomic i8 0, ptr @__profc_mcdc_tsan monotonic, align 1, !nosanitize
; COND:          %pgocount = load atomic i8, ptr @__profc_mcdc_tsan monotonic, align 1, !nosanitize
; COND-NEXT:     %pgocount.ifnonzero = icmp ne i8 %pgocount, 0
; COND-NEXT:     br i1 %pgocount.ifnonzero, label %[[COVER:.+]], label %[[CONT:.+]]
; COND:        [[COVER]]:
; COND-NEXT:     store atomic i8 0, ptr @__profc_mcdc_tsan monotonic, align 1, !nosanitize
; COND-NEXT:     br label %[[CONT]]
; COND:        [[CONT]]:
; CHECK:         %[[ADDR:.+]] = getelementptr inbounds i8, ptr @__profbm_mcdc_tsan, i32 %{{.+}}
; CHECK:         %[[SHIFTED:.+]] = shl i8 1, %{{.+}}
; CHECK-NEXT:    %[[BITS:.+]] = load atomic i8, ptr %[[ADDR]] monotonic, align 1, !nosanitize
; CHECK-NEXT:    %[[MASKED:.+]] = and i8 %[[BITS]], %[[SHIFTED]]
; CHECK-NEXT:    %[[SHOULDWRITE:.+]] = icmp ne i8 %[[MASKED]], %[[SHIFTED]]
; CHECK-NEXT:    br i1 %[[SHOULDWRITE]], label %[[WRITE:.+]], label %[[SKIP:.+]], !prof
; CHECK:       [[WRITE]]:
; CHECK-NEXT:    atomicrmw or ptr %[[ADDR]], i8 %[[SHIFTED]] monotonic, align 1, !nosanitize
;
entry:
  %A.addr = alloca i32, align 4
  %mcdc.addr = alloca i32, align 4
  call void @llvm.instrprof.cover(ptr @__profn_mcdc_tsan, i64 99278, i32 5, i32 0)
  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_mcdc_tsan, i64 99278, i32 1)
  store i32 0, ptr %mcdc.addr, align 4
  %0 = load i32, ptr %A.addr, align 4
  %tobool = icmp ne i32 %0, 0
  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @__profn_mcdc_tsan, i64 99278, i32 0, ptr %mcdc.addr)
  ret void
}

define dso_local void @mcdc_no_tsan(i32 noundef %A) {
; CHECK-LABEL: define dso_local void @mcdc_no_tsan(
; NOCOND:        store i8 0, ptr @__profc_mcdc_no_tsan, align 1, !nosanitize
; COND:          %pgocount = load i8, ptr @__profc_mcdc_no_tsan, align 1, !nosanitize
; COND-NEXT:     %pgocount.ifnonzero = icmp ne i8 %pgocount, 0
; COND-NEXT:     br i1 %pgocount.ifnonzero, label %[[COVER:.+]], label %[[CONT:.+]]
; COND:        [[COVER]]:
; COND-NEXT:     store i8 0, ptr @__profc_mcdc_no_tsan, align 1, !nosanitize
; COND-NEXT:     br label %[[CONT]]
; COND:        [[CONT]]:
; CHECK:         %[[ADDR:.+]] = getelementptr inbounds i8, ptr @__profbm_mcdc_no_tsan, i32 %{{.+}}
; CHECK:         %[[SHIFTED:.+]] = shl i8 1, %{{.+}}
; CHECK-NEXT:    %[[BITS:.+]] = load i8, ptr %[[ADDR]], align 1, !nosanitize
; CHECK-NEXT:    %[[OR:.+]] = or i8 %[[BITS]], %[[SHIFTED]]
; CHECK-NEXT:    store i8 %[[OR]], ptr %[[ADDR]], align 1, !nosanitize
;
entry:
  %A.addr = alloca i32, align 4
  %mcdc.addr = alloca i32, align 4
  call void @llvm.instrprof.cover(ptr @__profn_mcdc_no_tsan, i64 99278, i32 5, i32 0)
  call void @llvm.instrprof.mcdc.parameters(ptr @__profn_mcdc_no_tsan, i64 99278, i32 1)
  store i32 0, ptr %mcdc.addr, align 4
  %0 = load i32, ptr %A.addr, align 4
  %tobool = icmp ne i32 %0, 0
  call void @llvm.instrprof.mcdc.tvbitmap.update(ptr @__profn_mcdc_no_tsan, i64 99278, i32 0, ptr %mcdc.addr)
  ret void
}
