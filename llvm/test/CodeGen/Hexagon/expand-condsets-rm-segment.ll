; RUN: llc -march=hexagon -O2 < %s
; REQUIRES: asserts

target triple = "hexagon-unknown--elf"

%struct.cpumask = type { [1 x i32] }
%struct.load_weight = type { i32, i32 }

@sysctl_sched_latency = global i32 6000000, align 4
@normalized_sysctl_sched_latency = global i32 6000000, align 4
@sysctl_sched_tunable_scaling = global i8 1, align 1
@sysctl_sched_min_granularity = global i32 750000, align 4
@normalized_sysctl_sched_min_granularity = global i32 750000, align 4
@sysctl_sched_wakeup_granularity = global i32 1000000, align 4
@normalized_sysctl_sched_wakeup_granularity = global i32 1000000, align 4
@sysctl_sched_migration_cost = constant i32 500000, align 4
@sysctl_sched_shares_window = global i32 10000000, align 4
@sysctl_sched_child_runs_first = common global i32 0, align 4
@cpu_online_mask = external constant ptr

; Function Attrs: noinline nounwind
define void @sched_init_granularity() #0 {
entry:
  tail call fastcc void @update_sysctl()
  ret void
}

; Function Attrs: noinline nounwind
define internal fastcc void @update_sysctl() #0 {
entry:
  %call = tail call i32 @get_update_sysctl_factor()
  %0 = load i32, ptr @normalized_sysctl_sched_min_granularity, align 4, !tbaa !1
  %mul = mul i32 %0, %call
  store i32 %mul, ptr @sysctl_sched_min_granularity, align 4, !tbaa !1
  %1 = load i32, ptr @normalized_sysctl_sched_latency, align 4, !tbaa !1
  %mul1 = mul i32 %1, %call
  store i32 %mul1, ptr @sysctl_sched_latency, align 4, !tbaa !1
  %2 = load i32, ptr @normalized_sysctl_sched_wakeup_granularity, align 4, !tbaa !1
  %mul2 = mul i32 %2, %call
  store i32 %mul2, ptr @sysctl_sched_wakeup_granularity, align 4, !tbaa !1
  ret void
}

; Function Attrs: noinline nounwind
define i32 @calc_delta_mine(i32 %delta_exec, i32 %weight, ptr nocapture %lw) #0 {
entry:
  %cmp = icmp ugt i32 %weight, 1
  %conv = zext i32 %delta_exec to i64
  br i1 %cmp, label %if.then, label %if.end, !prof !5

if.then:                                          ; preds = %entry
  %conv2 = zext i32 %weight to i64
  %mul = mul i64 %conv2, %conv
  br label %if.end

if.end:                                           ; preds = %entry, %if.then
  %tmp.0 = phi i64 [ %mul, %if.then ], [ %conv, %entry ]
  %inv_weight = getelementptr inbounds %struct.load_weight, ptr %lw, i32 0, i32 1
  %0 = load i32, ptr %inv_weight, align 4, !tbaa !6
  %tobool4 = icmp eq i32 %0, 0
  br i1 %tobool4, label %if.then5, label %if.end22

if.then5:                                         ; preds = %if.end
  %1 = load i32, ptr %lw, align 4, !tbaa !9
  %lnot9 = icmp eq i32 %1, 0
  br i1 %lnot9, label %if.then17, label %if.else19, !prof !10

if.then17:                                        ; preds = %if.then5
  store i32 -1, ptr %inv_weight, align 4, !tbaa !6
  br label %if.end22

if.else19:                                        ; preds = %if.then5
  %div = udiv i32 -1, %1
  store i32 %div, ptr %inv_weight, align 4, !tbaa !6
  br label %if.end22

if.end22:                                         ; preds = %if.end, %if.then17, %if.else19
  %2 = phi i32 [ %0, %if.end ], [ -1, %if.then17 ], [ %div, %if.else19 ]
  %cmp23 = icmp ugt i64 %tmp.0, 4294967295
  br i1 %cmp23, label %if.then31, label %if.else37, !prof !10

if.then31:                                        ; preds = %if.end22
  %add = add i64 %tmp.0, 32768
  %shr = lshr i64 %add, 16
  %conv33 = zext i32 %2 to i64
  %mul34 = mul i64 %conv33, %shr
  %add35 = add i64 %mul34, 32768
  %shr36 = lshr i64 %add35, 16
  br label %if.end43

if.else37:                                        ; preds = %if.end22
  %conv39 = zext i32 %2 to i64
  %mul40 = mul i64 %conv39, %tmp.0
  %add41 = add i64 %mul40, 2147483648
  %shr42 = lshr i64 %add41, 32
  br label %if.end43

if.end43:                                         ; preds = %if.else37, %if.then31
  %tmp.1 = phi i64 [ %shr36, %if.then31 ], [ %shr42, %if.else37 ]
  %cmp49 = icmp ult i64 %tmp.1, 2147483647
  %3 = trunc i64 %tmp.1 to i32
  %conv51 = select i1 %cmp49, i32 %3, i32 2147483647
  ret i32 %conv51
}

declare i32 @get_update_sysctl_factor() #0
declare i32 @__bitmap_weight(ptr, i32) #0

attributes #0 = { noinline nounwind }

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!"branch_weights", i32 64, i32 4}
!6 = !{!7, !8, i64 4}
!7 = !{!"load_weight", !8, i64 0, !8, i64 4}
!8 = !{!"long", !3, i64 0}
!9 = !{!7, !8, i64 0}
!10 = !{!"branch_weights", i32 4, i32 64}
!11 = !{!12, !12, i64 0}
!12 = !{!"any pointer", !3, i64 0}
!13 = !{!3, !3, i64 0}
!14 = !{i32 45854, i32 45878}
