; RUN: opt -passes='loop-trap-analysis' --use-bounds-safety-traps-only -pass-remarks-missed='loop-trap-analysis' -disable-output -pass-remarks-output=%t.opt.yaml %s
; RUN: FileCheck --check-prefixes OPT-REM  --input-file=%t.opt.yaml %s

; REQUIRES: apple-disclosure-ios

; OPT-REM: --- !Analysis
; OPT-REM-NEXT: Pass:            loop-trap-analysis
; OPT-REM-NEXT: Name:            LoopTrap
; OPT-REM-NEXT: Function:        write_checks
; OPT-REM-NEXT: Args:
; OPT-REM-NEXT:   - String:          'Loop: '
; OPT-REM-NEXT:   - String:          for.body
; OPT-REM-NEXT:   - String:          ' '
; OPT-REM-NEXT:   - String:          "cannot be hoisted: \n"
; OPT-REM-NEXT:   - String:           |
; OPT-REM-NEXT: {{^[ 	]+$}}
; OPT-REM-NEXT:       The following instructions have side effects:
; OPT-REM-EMPTY: 
; OPT-REM-NEXT:   - String:          '	'
; OPT-REM-NEXT:   - String:          '  store i32 1, ptr %ptr.ind, align 4'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          "Reason:\n"
; OPT-REM-NEXT:   - String:          "Instruction may write to memory\n"
; OPT-REM-NEXT: ...
; OPT-REM-NEXT: --- !Analysis
; OPT-REM-NEXT: Pass:            loop-trap-analysis
; OPT-REM-NEXT: Name:            LoopTrapSummary
; OPT-REM-NEXT: Function:        write_checks
; OPT-REM-NEXT: Args:
; OPT-REM-NEXT:   - String:          "Trap checks results:\n"
; OPT-REM-NEXT:   - String:          'Total count of loops with traps '
; OPT-REM-NEXT:   - TotalCount:      '1'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          'Loops that maybe can be hoisted: '
; OPT-REM-NEXT:   - CountHoist:      '0'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          'Loops that cannot be hoisted: '
; OPT-REM-NEXT:   - CountCannotHoist: '1'
; OPT-REM-NEXT:   - String:          "\n"
define void @write_checks(ptr %base, i32 %N) {
entry:
  %ptr.lb = getelementptr i32, ptr %base, i32 0
  %ptr.ub = getelementptr i32, ptr %base, i32 %N 
  %cmp9.not = icmp eq i32 %N, 0
  br i1 %cmp9.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond.cleanup:                                 ; preds = %cont6, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %cont6
  %indvars.iv = phi i32 [ 0, %for.body.preheader ], [ %indvars.iv.next, %cont6 ]
  %ptr.ind = getelementptr i32, ptr %base, i32 %indvars.iv
  %cmp.ult = icmp ult ptr %ptr.ind, %ptr.ub, !annotation !1
  %cmp.uge = icmp uge ptr %ptr.ind, %ptr.lb, !annotation !2
  %or.cond = and i1 %cmp.ult, %cmp.uge, !annotation !2
  br i1 %or.cond, label %cont6, label %trap, !annotation !1

trap:                                             ; preds = %for.body
  tail call void @llvm.ubsantrap(i8 25), !annotation !3
  unreachable, !annotation !3

cont6:                                            ; preds = %for.body
  store i32 1, ptr %ptr.ind, align 4
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond.not = icmp eq i32 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}

; OPT-REM: --- !Analysis
; OPT-REM-NEXT: Pass:            loop-trap-analysis
; OPT-REM-NEXT: Name:            LoopTrap
; OPT-REM-NEXT: Function:        accumulate_checks
; OPT-REM-NEXT: Args:
; OPT-REM-NEXT:   - String:          'Loop: '
; OPT-REM-NEXT:   - String:          for.body
; OPT-REM-NEXT:   - String:          ' '
; OPT-REM-NEXT:   - String:          "can be hoisted\n"
; OPT-REM-NEXT: ...
; OPT-REM-NEXT: --- !Analysis
; OPT-REM-NEXT: Pass:            loop-trap-analysis
; OPT-REM-NEXT: Name:            LoopTrapSummary
; OPT-REM-NEXT: Function:        accumulate_checks
; OPT-REM-NEXT: Args:
; OPT-REM-NEXT:   - String:          "Trap checks results:\n"
; OPT-REM-NEXT:   - String:          'Total count of loops with traps '
; OPT-REM-NEXT:   - TotalCount:      '1'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          'Loops that maybe can be hoisted: '
; OPT-REM-NEXT:   - CountHoist:      '1'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          'Loops that cannot be hoisted: '
; OPT-REM-NEXT:   - CountCannotHoist: '0'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT: ...
define void @accumulate_checks(ptr %base, i32 %N) {
entry:
  %ptr.lb = getelementptr i32, ptr %base, i32 0
  %ptr.ub = getelementptr i32, ptr %base, i32 %N 
  %cmp9.not = icmp eq i32 %N, 0
  br i1 %cmp9.not, label %for.cond.cleanup, label %for.body.preheader

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond:                                         ; preds = %for.body
  %indvars.iv.next = add nuw nsw i32 %indvars.iv, 1
  %exitcond.not = icmp eq i32 %indvars.iv.next, %N
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body 

for.cond.cleanup:                                 ; preds = %for.cond, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.cond
  %indvars.iv = phi i32 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.cond ]
  %ptr.ind = getelementptr i32, ptr %base, i32 %indvars.iv
  %cmp.ult = icmp ult ptr %ptr.ind, %ptr.ub, !annotation !1
  %cmp.uge = icmp uge ptr %ptr.ind, %ptr.lb, !annotation !2
  %or.cond = and i1 %cmp.ult, %cmp.uge, !annotation !2
  br i1 %or.cond, label %for.cond, label %trap, !annotation !1

trap:                                             ; preds = %for.body
  tail call void @llvm.ubsantrap(i8 25), !annotation !3
  unreachable, !annotation !3
}

; OPT-REM: --- !Analysis
; OPT-REM-NEXT: Pass:            loop-trap-analysis
; OPT-REM-NEXT: Name:            LoopTrap
; OPT-REM-NEXT: Function:        trip_count_unknown
; OPT-REM-NEXT: Args:
; OPT-REM-NEXT:   - String:          'Loop: '
; OPT-REM-NEXT:   - String:          loop
; OPT-REM-NEXT:   - String:          ' '
; OPT-REM-NEXT:   - String:          "cannot be hoisted: \n"
; OPT-REM-NEXT:   - String:          "Backedge is not computable.\n"
; OPT-REM-NEXT: ...
; OPT-REM-NEXT: --- !Analysis
; OPT-REM-NEXT: Pass:            loop-trap-analysis
; OPT-REM-NEXT: Name:            LoopTrapSummary
; OPT-REM-NEXT: Function:        trip_count_unknown
; OPT-REM-NEXT: Args:
; OPT-REM-NEXT:   - String:          "Trap checks results:\n"
; OPT-REM-NEXT:   - String:          'Total count of loops with traps '
; OPT-REM-NEXT:   - TotalCount:      '1'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          'Loops that maybe can be hoisted: '
; OPT-REM-NEXT:   - CountHoist:      '0'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT:   - String:          'Loops that cannot be hoisted: '
; OPT-REM-NEXT:   - CountCannotHoist: '1'
; OPT-REM-NEXT:   - String:          "\n"
; OPT-REM-NEXT: ...
define void @trip_count_unknown(ptr %A, ptr %B, i32 %N, i32 %M) {
entry:
  %cmp37.not = icmp eq i32 %N, 0
  br i1 %cmp37.not, label %exit, label %for.body.lr.ph

for.body.lr.ph:                                   ; preds = %entry
  %idx.ext3 = zext i32 %M to i64
  %add.ptr4 = getelementptr inbounds i32, ptr %B, i64 %idx.ext3
  %wide.trip.count = zext i32 %N to i64
  br label %loop

loop:                                         
  %indvars.iv = phi i64 [ 0, %for.body.lr.ph ], [ %indvars.iv.next, %next ]
  %a.iv.next = getelementptr i32, ptr %A, i64 %indvars.iv
  %b.iv.next = getelementptr i32, ptr %B, i64 %indvars.iv
  %cond = icmp ule i64 %indvars.iv, %wide.trip.count
  %cmp.b.ult = icmp ult ptr %b.iv.next, %add.ptr4, !annotation !1
  %cmp.b.uge = icmp uge ptr %b.iv.next, %B, !annotation !2
  %or.cond = and i1 %cmp.b.ult, %cmp.b.uge, !annotation !2
  %b.at.i = load i32, ptr %b.iv.next
  %loop.cond = icmp eq i32 %b.at.i, 0
  br i1 %loop.cond, label %cond1, label %exit

cond1:                                           ; preds = %for.body
  br i1 %or.cond, label %trap, label %next, !annotation !2

next: 
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  br i1 %loop.cond, label %loop, label %exit

trap:                                             
  tail call void @llvm.ubsantrap(i8 25), !annotation !3
  unreachable, !annotation !3

exit:                                          
  ret void
}

declare void @llvm.ubsantrap(i8 immarg) 

!1 = !{!"bounds-safety-check-ptr-lt-upper-bound"}
!2 = !{!"bounds-safety-check-ptr-ge-lower-bound"}
!3 = !{!"bounds-safety-check-ptr-lt-upper-bound", !"bounds-safety-check-ptr-ge-lower-bound"}

; OPT-REM-NOT: --- !Analysis
