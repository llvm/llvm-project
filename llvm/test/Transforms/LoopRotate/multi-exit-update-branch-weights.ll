; RUN: opt < %s -passes='loop-rotate<update-branch-weights>' -S | FileCheck %s
;
; Multi-exit loop reduced from a real PGO trace (treeup.c::update_tree).
; The loop has two exits (header and body). Loop-rotate previously assumed
; preheader_entries == header_exit_count and derived EnterWeight from the
; wrong edge, producing inconsistent weights on the rotated guard and latch.
;
; With the fix, EnterWeight is derived from the preheader entry count, so
; rotated weights remain consistent with the original iteration profile.
;
; Original header (TEST):
;   br i1 %cmp18, label %CONTINUE, label %if.end20
;     !prof !51                   ; {1977246, 82880387}
; Preheader edge (RECURSION -> TEST.preheader): 31075547
;
; After rotation:
;   Guard (TEST.preheader):
;     !prof {1, 31075546}
;   Latch (if.end23):
;     !prof {1977245, 51804841}

source_filename = "treeup.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
%struct.arc = type { i32, i64, ptr, ptr, i16, ptr, ptr, i64, i64 }
%struct.node = type { i64, i32, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, i64, i64, i32, i32 }
; Function Attrs: nounwind uwtable
define dso_local void @update_tree(i64 noundef %cycle_ori, i64 noundef %new_orientation, i64 noundef %delta, i64 noundef %new_flow, ptr noundef %iplus, ptr noundef %jplus, ptr noundef captures(address) %iminus, ptr noundef captures(address) %jminus, ptr noundef readnone captures(
address) %w, ptr noundef %bea, i64 noundef %sigma, i64 noundef %feas_tol) local_unnamed_addr #0 !prof !34 {
entry:
  %tail = getelementptr inbounds nuw %struct.arc, ptr %bea, i64 0, i32 2
  %0 = load ptr, ptr %tail, align 8
  %cmp = icmp eq ptr %0, %jplus
  %cmp1 = icmp slt i64 %sigma, 0
  %or.cond = and i1 %cmp1, %cmp
  br i1 %or.cond, label %if.then, label %lor.lhs.false, !prof !45
lor.lhs.false:                                    ; preds = %entry
  %cmp3 = icmp eq ptr %0, %iplus
  %cmp5 = icmp sgt i64 %sigma, 0
  %or.cond224 = and i1 %cmp5, %cmp3
  br i1 %or.cond224, label %if.then, label %if.else, !prof !46
if.then:                                          ; preds = %lor.lhs.false, %entry
  %cond = tail call i64 @llvm.abs.i64(i64 %sigma, i1 true)
  br label %if.end
if.else:                                          ; preds = %lor.lhs.false
  %cond12 = tail call i64 @llvm.abs.i64(i64 %sigma, i1 true)
  %sub13 = sub nsw i64 0, %cond12
  br label %if.end
if.end:                                           ; preds = %if.else, %if.then
  %sigma.addr.0 = phi i64 [ %cond, %if.then ], [ %sub13, %if.else ]
  %1 = load i64, ptr %iminus, align 8
  %add = add nsw i64 %1, %sigma.addr.0
  store i64 %add, ptr %iminus, align 8
  br label %RECURSION
RECURSION:                                        ; preds = %ITERATION, %if.end
  %father.0 = phi ptr [ %iminus, %if.end ], [ %temp.0, %ITERATION ]
  %child = getelementptr inbounds nuw %struct.node, ptr %father.0, i64 0, i32 2
  %2 = load ptr, ptr %child, align 8
  %tobool.not = icmp eq ptr %2, null
  br i1 %tobool.not, label %TEST.preheader, label %ITERATION, !prof !50
TEST.preheader:                                   ; preds = %RECURSION
  br label %TEST
ITERATION.loopexit:                               ; preds = %if.end20
  %.lcssa = phi ptr [ %4, %if.end20 ]
  br label %ITERATION
ITERATION:                                        ; preds = %ITERATION.loopexit, %RECURSION
  %temp.0 = phi ptr [ %2, %RECURSION ], [ %.lcssa, %ITERATION.loopexit ]
  %3 = load i64, ptr %temp.0, align 8
  %add16 = add nsw i64 %3, %sigma.addr.0
  store i64 %add16, ptr %temp.0, align 8
  br label %RECURSION
TEST:                                             ; preds = %TEST.preheader, %if.end23
  %father.1 = phi ptr [ %5, %if.end23 ], [ %father.0, %TEST.preheader ]
  %cmp18 = icmp eq ptr %father.1, %iminus
  br i1 %cmp18, label %CONTINUE, label %if.end20, !prof !51
if.end20:                                         ; preds = %TEST
  %sibling = getelementptr inbounds nuw %struct.node, ptr %father.1, i64 0, i32 4
  %4 = load ptr, ptr %sibling, align 8
  %tobool21.not = icmp eq ptr %4, null
  br i1 %tobool21.not, label %if.end23, label %ITERATION.loopexit, !prof !53
if.end23:                                         ; preds = %if.end20
  %pred = getelementptr inbounds nuw %struct.node, ptr %father.1, i64 0, i32 3
  %5 = load ptr, ptr %pred, align 8
  br label %TEST
CONTINUE:                                         ; preds = %TEST
  %pred24 = getelementptr inbounds nuw %struct.node, ptr %iplus, i64 0, i32 3
  %6 = load ptr, ptr %pred24, align 8
  %depth = getelementptr inbounds nuw %struct.node, ptr %iminus, i64 0, i32 11
  %7 = load i64, ptr %depth, align 8
  br label %while.cond
while.cond:                                       ; preds = %if.end61, %CONTINUE
  %new_basic_arc.0 = phi ptr [ %bea, %CONTINUE ], [ %15, %if.end61 ]
  %father.2 = phi ptr [ %6, %CONTINUE ], [ %17, %if.end61 ]
  %temp.1 = phi ptr [ %iplus, %CONTINUE ], [ %father.2, %if.end61 ]
  %new_pred.0 = phi ptr [ %jplus, %CONTINUE ], [ %temp.1, %if.end61 ]
  %new_flow.addr.0 = phi i64 [ %new_flow, %CONTINUE ], [ %flow_temp.0, %if.end61 ]
  %new_orientation.addr.0 = phi i64 [ %new_orientation, %CONTINUE ], [ %conv, %if.end61 ]
  %new_depth.0 = phi i64 [ %7, %CONTINUE ], [ %sub68, %if.end61 ]
  %cmp25.not = icmp eq ptr %temp.1, %jminus
  br i1 %cmp25.not, label %while.end, label %while.body, !prof !56
while.body:                                       ; preds = %while.cond
  %sibling26 = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 4
  %8 = load ptr, ptr %sibling26, align 8
  %tobool27.not = icmp eq ptr %8, null
  br i1 %tobool27.not, label %if.end31, label %if.then28, !prof !57
if.then28:                                        ; preds = %while.body
  %sibling_prev = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 5
  %9 = load ptr, ptr %sibling_prev, align 8
  %sibling_prev30 = getelementptr inbounds nuw %struct.node, ptr %8, i64 0, i32 5
  store ptr %9, ptr %sibling_prev30, align 8
  br label %if.end31
if.end31:                                         ; preds = %if.then28, %while.body
  %sibling_prev32 = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 5
  %10 = load ptr, ptr %sibling_prev32, align 8
  %tobool33.not = icmp eq ptr %10, null
  br i1 %tobool33.not, label %if.else38, label %if.then34, !prof !59
if.then34:                                        ; preds = %if.end31
  %sibling37 = getelementptr inbounds nuw %struct.node, ptr %10, i64 0, i32 4
  store ptr %8, ptr %sibling37, align 8
  br label %if.end41
if.else38:                                        ; preds = %if.end31
  %child40 = getelementptr inbounds nuw %struct.node, ptr %father.2, i64 0, i32 2
  store ptr %8, ptr %child40, align 8
  br label %if.end41
if.end41:                                         ; preds = %if.else38, %if.then34
  %pred42 = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 3
  store ptr %new_pred.0, ptr %pred42, align 8
  %child43 = getelementptr inbounds nuw %struct.node, ptr %new_pred.0, i64 0, i32 2
  %11 = load ptr, ptr %child43, align 8
  store ptr %11, ptr %sibling26, align 8
  %tobool46.not = icmp eq ptr %11, null
  br i1 %tobool46.not, label %if.end50, label %if.then47, !prof !60
if.then47:                                        ; preds = %if.end41
  %sibling_prev49 = getelementptr inbounds nuw %struct.node, ptr %11, i64 0, i32 5
  store ptr %temp.1, ptr %sibling_prev49, align 8
  br label %if.end50
if.end50:                                         ; preds = %if.then47, %if.end41
  store ptr %temp.1, ptr %child43, align 8
  store ptr null, ptr %sibling_prev32, align 8
  %orientation = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 1
  %12 = load i32, ptr %orientation, align 8
  %tobool53.not = icmp eq i32 %12, 0
  %conv = zext i1 %tobool53.not to i64
  %cmp54 = icmp eq i64 %cycle_ori, %conv
  br i1 %cmp54, label %if.then56, label %if.else58, !prof !62
if.then56:                                        ; preds = %if.end50
  %flow = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 10
  %13 = load i64, ptr %flow, align 8
  %add57 = add nsw i64 %13, %delta
  br label %if.end61
if.else58:                                        ; preds = %if.end50
  %flow59 = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 10
  %14 = load i64, ptr %flow59, align 8
  %sub60 = sub nsw i64 %14, %delta
  br label %if.end61
if.end61:                                         ; preds = %if.else58, %if.then56
  %flow_temp.0 = phi i64 [ %add57, %if.then56 ], [ %sub60, %if.else58 ]
  %basic_arc = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 6
  %15 = load ptr, ptr %basic_arc, align 8
  %depth62 = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 11
  %16 = load i64, ptr %depth62, align 8
  %conv63 = trunc i64 %new_orientation.addr.0 to i32
  store i32 %conv63, ptr %orientation, align 8
  %flow65 = getelementptr inbounds nuw %struct.node, ptr %temp.1, i64 0, i32 10
  store i64 %new_flow.addr.0, ptr %flow65, align 8
  store ptr %new_basic_arc.0, ptr %basic_arc, align 8
  store i64 %new_depth.0, ptr %depth62, align 8
  %sub68 = sub nsw i64 %7, %16
  %pred69 = getelementptr inbounds nuw %struct.node, ptr %father.2, i64 0, i32 3
  %17 = load ptr, ptr %pred69, align 8
  br label %while.cond, !llvm.loop !65
while.end:                                        ; preds = %while.cond
  %cmp70 = icmp sgt i64 %delta, %feas_tol
  br i1 %cmp70, label %for.cond.preheader, label %for.cond110.preheader, !prof !67
for.cond110.preheader:                            ; preds = %while.end
  br label %for.cond110
for.cond.preheader:                               ; preds = %while.end
  br label %for.cond
for.cond:                                         ; preds = %for.cond.preheader, %for.inc
  %temp.2 = phi ptr [ %22, %for.inc ], [ %jminus, %for.cond.preheader ]
  %cmp73.not = icmp eq ptr %temp.2, %w
  br i1 %cmp73.not, label %for.cond89.preheader, label %for.body, !prof !68
for.cond89.preheader:                             ; preds = %for.cond
  br label %for.cond89
for.body:                                         ; preds = %for.cond
  %depth75 = getelementptr inbounds nuw %struct.node, ptr %temp.2, i64 0, i32 11
  %18 = load i64, ptr %depth75, align 8
  %sub76 = sub nsw i64 %18, %7
  store i64 %sub76, ptr %depth75, align 8
  %orientation77 = getelementptr inbounds nuw %struct.node, ptr %temp.2, i64 0, i32 1
  %19 = load i32, ptr %orientation77, align 8
  %conv78 = sext i32 %19 to i64
  %cmp79.not = icmp eq i64 %cycle_ori, %conv78
  br i1 %cmp79.not, label %if.else84, label %if.then81
if.then81:                                        ; preds = %for.body
  %flow82 = getelementptr inbounds nuw %struct.node, ptr %temp.2, i64 0, i32 10
  %20 = load i64, ptr %flow82, align 8
  %add83 = add nsw i64 %20, %delta
  store i64 %add83, ptr %flow82, align 8
  br label %for.inc
if.else84:                                        ; preds = %for.body
  %flow85 = getelementptr inbounds nuw %struct.node, ptr %temp.2, i64 0, i32 10
  %21 = load i64, ptr %flow85, align 8
  %sub86 = sub nsw i64 %21, %delta
  store i64 %sub86, ptr %flow85, align 8
  br label %for.inc
for.inc:                                          ; preds = %if.then81, %if.else84
  %pred88 = getelementptr inbounds nuw %struct.node, ptr %temp.2, i64 0, i32 3
  %22 = load ptr, ptr %pred88, align 8
  br label %for.cond, !llvm.loop !69
for.cond89:                                       ; preds = %for.cond89.preheader, %for.inc106
  %temp.3 = phi ptr [ %27, %for.inc106 ], [ %jplus, %for.cond89.preheader ]
  %cmp90.not = icmp eq ptr %temp.3, %w
  br i1 %cmp90.not, label %if.end128.loopexit, label %for.body92, !prof !68
for.body92:                                       ; preds = %for.cond89
  %depth93 = getelementptr inbounds nuw %struct.node, ptr %temp.3, i64 0, i32 11
  %23 = load i64, ptr %depth93, align 8
  %add94 = add nsw i64 %23, %7
  store i64 %add94, ptr %depth93, align 8
  %orientation95 = getelementptr inbounds nuw %struct.node, ptr %temp.3, i64 0, i32 1
  %24 = load i32, ptr %orientation95, align 8
  %conv96 = sext i32 %24 to i64
  %cmp97 = icmp eq i64 %cycle_ori, %conv96
  br i1 %cmp97, label %if.then99, label %if.else102
if.then99:                                        ; preds = %for.body92
  %flow100 = getelementptr inbounds nuw %struct.node, ptr %temp.3, i64 0, i32 10
  %25 = load i64, ptr %flow100, align 8
  %add101 = add nsw i64 %25, %delta
  store i64 %add101, ptr %flow100, align 8
  br label %for.inc106
if.else102:                                       ; preds = %for.body92
  %flow103 = getelementptr inbounds nuw %struct.node, ptr %temp.3, i64 0, i32 10
  %26 = load i64, ptr %flow103, align 8
  %sub104 = sub nsw i64 %26, %delta
  store i64 %sub104, ptr %flow103, align 8
  br label %for.inc106
for.inc106:                                       ; preds = %if.then99, %if.else102
  %pred107 = getelementptr inbounds nuw %struct.node, ptr %temp.3, i64 0, i32 3
  %27 = load ptr, ptr %pred107, align 8
  br label %for.cond89, !llvm.loop !70
for.cond110:                                      ; preds = %for.cond110.preheader, %for.body113
  %temp.4 = phi ptr [ %29, %for.body113 ], [ %jminus, %for.cond110.preheader ]
  %cmp111.not = icmp eq ptr %temp.4, %w
  br i1 %cmp111.not, label %for.cond119.preheader, label %for.body113, !prof !71
for.cond119.preheader:                            ; preds = %for.cond110
  br label %for.cond119
for.body113:                                      ; preds = %for.cond110
  %depth114 = getelementptr inbounds nuw %struct.node, ptr %temp.4, i64 0, i32 11
  %28 = load i64, ptr %depth114, align 8
  %sub115 = sub nsw i64 %28, %7
  store i64 %sub115, ptr %depth114, align 8
  %pred117 = getelementptr inbounds nuw %struct.node, ptr %temp.4, i64 0, i32 3
  %29 = load ptr, ptr %pred117, align 8
  br label %for.cond110, !llvm.loop !72
for.cond119:                                      ; preds = %for.cond119.preheader, %for.body122
  %temp.5 = phi ptr [ %31, %for.body122 ], [ %jplus, %for.cond119.preheader ]
  %cmp120.not = icmp eq ptr %temp.5, %w
  br i1 %cmp120.not, label %if.end128.loopexit225, label %for.body122, !prof !73
for.body122:                                      ; preds = %for.cond119
  %depth123 = getelementptr inbounds nuw %struct.node, ptr %temp.5, i64 0, i32 11
  %30 = load i64, ptr %depth123, align 8
  %add124 = add nsw i64 %30, %7
  store i64 %add124, ptr %depth123, align 8
  %pred126 = getelementptr inbounds nuw %struct.node, ptr %temp.5, i64 0, i32 3
  %31 = load ptr, ptr %pred126, align 8
  br label %for.cond119, !llvm.loop !74
if.end128.loopexit:                               ; preds = %for.cond89
  br label %if.end128
if.end128.loopexit225:                            ; preds = %for.cond119
  br label %if.end128
if.end128:                                        ; preds = %if.end128.loopexit225, %if.end128.loopexit
  ret void
}
; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.abs.i64(i64, i1 immarg) #1

!llvm.module.flags = !{!80}
!80 = !{i32 1, !"ProfileSummary", !81}
!81 = !{!82, !83, !84, !85, !86, !87, !88, !89}
!82 = !{!"ProfileFormat", !"InstrProf"}
!83 = !{!"TotalCount", i64 84857633}
!84 = !{!"MaxCount", i64 82880387}
!85 = !{!"MaxInternalCount", i64 82880387}
!86 = !{!"MaxFunctionCount", i64 82880387}
!87 = !{!"NumCounts", i64 64}
!88 = !{!"NumFunctions", i64 1}
!89 = !{!"DetailedSummary", !90}
!90 = !{!91, !92, !93}
!91 = !{i32 10000, i64 82880387, i32 1}
!92 = !{i32 999000, i64 53782086, i32 1}
!93 = !{i32 999999, i64 1977246, i32 1}

!34 = !{!"function_entry_count", i64 1977246}
!45 = !{!"branch_weights", i32 9915, i32 1967331}
!46 = !{!"branch_weights", i32 6597, i32 1960734}
!50 = !{!"branch_weights", i32 31075547, i32 53782086}
!51 = !{!"branch_weights", i32 1977246, i32 82880387}
!53 = !{!"branch_weights", i32 53782086, i32 29098301}
!56 = !{!"branch_weights", i32 1977246, i32 2385079}
!57 = !{!"branch_weights", i32 746895, i32 1638184}
!59 = !{!"branch_weights", i32 791318, i32 1593761}
!60 = !{!"branch_weights", i32 476345, i32 1908734}
!62 = !{!"branch_weights", i32 240086, i32 2144993}
!65 = distinct !{!65, !66}
!66 = !{!"llvm.loop.mustprogress"}
!67 = !{!"branch_weights", i32 16512, i32 1960734}
!68 = !{!"branch_weights", i32 16512, i32 0}
!69 = distinct !{!69, !66}
!70 = distinct !{!70, !66}
!71 = !{!"branch_weights", i32 1960734, i32 110585854}
!72 = distinct !{!72, !66}
!73 = !{!"branch_weights", i32 1960734, i32 119591748}
!74 = distinct !{!74, !66}

; CHECK-LABEL: define dso_local void @update_tree(

; Rotated guard at the new preheader picks EnterWeight from the actual
; preheader edge weight (31075547) rather than the header's loop-edge weight.
; CHECK:      TEST.preheader:
; CHECK:        br i1 %cmp{{[0-9]+}}, label %TEST.preheader.CONTINUE_crit_edge, label %if.end20.lr.ph, !prof [[GUARD:![0-9]+]]

; Rotated latch carries the leftover of the original header weights.
; CHECK:      if.end23:
; CHECK:        br i1 %cmp{{[0-9]+}}, label %TEST.CONTINUE_crit_edge, label %if.end20, !prof [[LATCH:![0-9]+]]

; CHECK-DAG: [[GUARD]] = !{!"branch_weights", i32 1, i32 31075546}
; CHECK-DAG: [[LATCH]] = !{!"branch_weights", i32 1977245, i32 51804841}
