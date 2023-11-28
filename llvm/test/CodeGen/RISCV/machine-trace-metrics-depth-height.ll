; RUN: llc -mtriple=riscv64 -mattr=+m,+v -debug-only=machine-trace-metrics %s -o /dev/null 2>&1 | FileCheck %s

@heap_ptr = global ptr null, align 8
@heap_end = global ptr null, align 8
@heap_requested = global i64 0, align 8

define ptr @mtm_depth_height(ptr %ptr, i64 %size) {
; CHECK:      Computing MinInstr trace through [[BB1:%bb.[0-9]+]]
; CHECK-NEXT:   pred for [[BB0:%bb.[0-9]+]]: null
; CHECK-NEXT:   pred for [[BB1]]: [[BB0]]
; CHECK-NEXT:   succ for [[BB10:%bb.[0-9]+]]: null
; CHECK-NEXT:   succ for [[BB14:%bb.[0-9]+]]: [[BB10]]
; CHECK-NEXT:   succ for [[BB15:%bb.[0-9]+]]: [[BB10]]
; CHECK-NEXT:   succ for [[BB9:%bb.[0-9]+]]: null
; CHECK-NEXT:   succ for [[BB8:%bb.[0-9]+]]: [[BB9]]
; CHECK-NEXT:   succ for [[BB16:%bb.[0-9]+]]: [[BB8]]
; CHECK-NEXT:   succ for [[BB17:%bb.[0-9]+]]: [[BB8]]
; CHECK-NEXT:   succ for [[BB6:%bb.[0-9]+]]: null
; CHECK-NEXT:   succ for [[BB5:%bb.[0-9]+]]: [[BB6]]
; CHECK-NEXT:   succ for [[BB4:%bb.[0-9]+]]: [[BB5]]
; CHECK-NEXT:   succ for [[BB12:%bb.[0-9]+]]: [[BB16]]
; CHECK-NEXT:   succ for [[BB11:%bb.[0-9]+]]: [[BB12]]
; CHECK-NEXT:   succ for [[BB3:%bb.[0-9]+]]: [[BB11]]
; CHECK-NEXT:   succ for [[BB2:%bb.[0-9]+]]: [[BB15]]
; CHECK-NEXT:   succ for [[BB1]]: [[BB14]]
; CHECK-EMPTY:
; CHECK-NEXT: Depths for [[BB0]]:
; CHECK-NEXT:       0 Instructions
; CHECK-NEXT: 0	[[R22:%[0-9]+]]:gpr = COPY [[X11:\$x[0-9]+]]
; CHECK-NEXT: 0	[[R21:%[0-9]+]]:gpr = COPY [[X10:\$x[0-9]+]]
; CHECK-NEXT: 0	[[R24:%[0-9]+]]:gpr = SLTIU [[R21]]:gpr, 1
; CHECK-NEXT: 0	[[R25:%[0-9]+]]:gpr = SLTIU [[R22]]:gpr, 1
; CHECK-NEXT: 1	[[R26:%[0-9]+]]:gpr = OR killed [[R24]]:gpr
; CHECK-NEXT: 2	BEQ killed [[R26]]:gpr, [[X0:\$x[0-9]+]], [[BB1]]
; CHECK-EMPTY:
; CHECK-NEXT: Depths for [[BB1]]:
; CHECK-NEXT:       4 Instructions
; CHECK-NEXT: 0	[[R29:%[0-9]+]]:gpr = LUI target-flags(riscv-hi) @heap_ptr
; CHECK-NEXT: 1	[[R0:%[0-9]+]]:gpr = LD [[R29]]:gpr, target-flags(riscv-lo) @heap_ptr 
; CHECK-NEXT: 5	[[R30:%[0-9]+]]:gpr = ADD [[R0]]:gpr, [[R22]]:gpr
; CHECK-NEXT: 0	[[R31:%[0-9]+]]:gpr = LUI target-flags(riscv-hi) @heap_requested
; CHECK-NEXT: 1	[[R32:%[0-9]+]]:gpr = LD [[R31]]:gpr, target-flags(riscv-lo) @heap_requested 
; CHECK-NEXT: 5	[[R33:%[0-9]+]]:gpr = ADD killed [[R32]]:gpr, [[R22]]:gpr
; CHECK-NEXT: 6	[[R34:%[0-9]+]]:gpr = ANDI [[R30]]:gpr, 7
; CHECK-NEXT: 7	[[R35:%[0-9]+]]:gpr = SLTIU [[R34]]:gpr, 1
; CHECK-NEXT: 0	[[R36:%[0-9]+]]:gpr = ADDI [[X0]], 8
; CHECK-NEXT: 7	[[R37:%[0-9]+]]:gpr = nuw nsw SUB killed [[R36]]:gpr, [[R34]]:gpr
; CHECK-NEXT: 8	[[R38:%[0-9]+]]:gpr = ADDI killed [[R35]]:gpr, -1
; CHECK-NEXT: 9	[[R39:%[0-9]+]]:gpr = AND killed [[R38]]:gpr, killed [[R37]]:gpr
; CHECK-NEXT: 10	[[R40:%[0-9]+]]:gpr = ADD killed [[R33]]:gpr, [[R39]]:gpr
; CHECK-NEXT: 10	[[R1:%[0-9]+]]:gpr = ADD [[R30]]:gpr, [[R39:%[0-9]+]]:gpr
; CHECK-NEXT: 1	SD killed [[R40]]:gpr, [[R31]]:gpr, target-flags(riscv-lo) @heap_requested 
; CHECK-NEXT: 0	[[R41:%[0-9]+]]:gpr = LUI target-flags(riscv-hi) @heap_end
; CHECK-NEXT: 1	[[R42:%[0-9]+]]:gpr = LD killed [[R41]]:gpr, target-flags(riscv-lo) @heap_end 
; CHECK-NEXT: 11	BGEU killed [[R42]]:gpr, [[R1]]:gpr, [[BB2]]
; CHECK-NEXT: Heights for [[BB10]]:
; CHECK-NEXT:       1 Instructions
; CHECK-NEXT: 0	PseudoRET implicit [[X10]]
; CHECK-NEXT: 0	[[X10]] = COPY [[R20:%[0-9]+]]:gpr
; CHECK-NEXT: 0	[[R20]]:gpr = PHI [[R23:%[0-9]+]]:gpr, [[BB13:%bb.[0-9]+]], [[R28:%[0-9]+]]:gpr, [[BB14]], [[R44:%[0-9]+]]:gpr, [[BB15]], [[R0]]:gpr, [[BB7:%bb.[0-9]+]], [[R0]]:gpr, [[BB9]]
; CHECK-NEXT: [[BB10]] Live-ins:
; CHECK-NEXT: Heights for [[BB14]]:
; CHECK-NEXT:       2 Instructions
; CHECK-NEXT: pred	0	[[R20]]:gpr = PHI [[R23]]:gpr, [[BB13]], [[R28]]:gpr, [[BB14]], [[R44]]:gpr, [[BB15]], [[R0]]:gpr, [[BB7]], [[R0]]:gpr, [[BB9]]
; CHECK-NEXT: 0	PseudoBR [[BB10]]
; CHECK-NEXT: 0	[[R28]]:gpr = COPY [[R43:%[0-9]+]]:gpr
; CHECK-NEXT: 0	[[R43]]:gpr = COPY [[X0]]
; CHECK-NEXT: [[BB14]] Live-ins: X0@0
; CHECK-NEXT: Heights for [[BB1]]:
; CHECK-NEXT:      20 Instructions
; CHECK-NEXT: 11	0	BGEU killed [[R42]]:gpr, [[R1]]:gpr, [[BB2]]
; CHECK-NEXT: 11	4	[[R42]]:gpr = LD killed [[R41]]:gpr, target-flags(riscv-lo) @heap_end 
; CHECK-NEXT: 11	5	[[R41]]:gpr = LUI target-flags(riscv-hi) @heap_end
; CHECK-NEXT: 11	0	SD killed [[R40]]:gpr, [[R31]]:gpr, target-flags(riscv-lo) @heap_requested 
; CHECK-NEXT: 11	1	[[R1]]:gpr = ADD [[R30]]:gpr, [[R39]]:gpr
; CHECK-NEXT: 11	0	[[R40]]:gpr = ADD killed [[R33]]:gpr, [[R39]]:gpr
; CHECK-NEXT: 11	2	[[R39]]:gpr = AND killed [[R38]]:gpr, killed [[R37]]:gpr
; CHECK-NEXT: 11	3	[[R38]]:gpr = ADDI killed [[R35]]:gpr, -1
; CHECK-NEXT: 11	3	[[R37]]:gpr = nuw nsw SUB killed [[R36]]:gpr, [[R34]]:gpr
; CHECK-NEXT: 11	4	[[R36]]:gpr = ADDI [[X0]], 8
; CHECK-NEXT: 11	4	[[R35]]:gpr = SLTIU [[R34]]:gpr, 1
; CHECK-NEXT: 11	5	[[R34]]:gpr = ANDI [[R30]]:gpr, 7
; CHECK-NEXT: 11	1	[[R33]]:gpr = ADD killed [[R32]]:gpr, [[R22]]:gpr
; CHECK-NEXT: 11	5	[[R32]]:gpr = LD [[R31]]:gpr, target-flags(riscv-lo) @heap_requested 
; CHECK-NEXT: 11	6	[[R31]]:gpr = LUI target-flags(riscv-hi) @heap_requested
; CHECK-NEXT: 11	6	[[R30]]:gpr = ADD [[R0]]:gpr, [[R22]]:gpr
; CHECK-NEXT: 11	10	[[R0]]:gpr = LD [[R29]]:gpr, target-flags(riscv-lo) @heap_ptr 
; CHECK-NEXT: 11	11	[[R29]]:gpr = LUI target-flags(riscv-hi) @heap_ptr
; CHECK-NEXT: [[BB1]] Live-ins: [[R22]]@6 X0@4
; CHECK-NEXT: Critical path: 11
entry:
  %ptrint = ptrtoint ptr %ptr to i64
  %cmp = icmp eq ptr %ptr, null
  %cmp.i = icmp eq i64 %size, 0
  %or.cond = or i1 %cmp, %cmp.i
  br i1 %or.cond, label %return, label %if.end

if.end:                                          ; preds = %entry
  %0 = load ptr, ptr @heap_ptr, align 8
  %1 = ptrtoint ptr %0 to i64
  %add.ptr = getelementptr inbounds i8, ptr %0, i64 %size
  %2 = load i64, ptr @heap_requested, align 8
  %add = add i64 %2, %size
  %3 = ptrtoint ptr %add.ptr to i64
  %rem = and i64 %3, 7
  %cmp.eq = icmp eq i64 %rem, 0
  %sub = sub nuw nsw i64 8, %rem
  %sel = select i1 %cmp.eq, i64 0, i64 %sub
  %storemerge = add i64 %add, %sel
  %next_heap_ptr = getelementptr inbounds i8, ptr %add.ptr, i64 %sel
  store i64 %storemerge, ptr @heap_requested, align 8
  %4 = load ptr, ptr @heap_end, align 8
  %cmp.ugt = icmp ugt ptr %next_heap_ptr, %4
  br i1 %cmp.ugt, label %return, label %exit

exit:                                            ; preds = %if.end
  store ptr %next_heap_ptr, ptr @heap_ptr, align 8
  %cmp.not = icmp eq ptr %0, null
  br i1 %cmp.not, label %return, label %ph

ph:                                              ; preds = %exit
  %5 = tail call i64 @llvm.vscale.i64()
  %6 = shl nuw nsw i64 %5, 4
  %7 = tail call i64 @llvm.umax.i64(i64 %6, i64 32)
  %min.iters.check = icmp ugt i64 %7, %size
  br i1 %min.iters.check, label %for.ph, label %vector.memcheck

vector.memcheck:                                  ; preds = %ph
  %8 = tail call i64 @llvm.vscale.i64()
  %9 = shl nuw nsw i64 %8, 4
  %10 = sub i64 %1, %ptrint
  %diff.check = icmp ult i64 %10, %9
  br i1 %diff.check, label %for.ph, label %vector.ph

vector.ph:                                        ; preds = %vector.memcheck
  %11 = tail call i64 @llvm.vscale.i64()
  %.neg = mul nsw i64 %11, -16
  %n.vec = and i64 %.neg, %size
  %12 = tail call i64 @llvm.vscale.i64()
  %13 = shl nuw nsw i64 %12, 4
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %index = phi i64 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %14 = getelementptr inbounds i8, ptr %ptr, i64 %index
  %wide.load = load <vscale x 16 x i8>, ptr %14, align 1
  %15 = getelementptr inbounds i8, ptr %0, i64 %index
  store <vscale x 16 x i8> %wide.load, ptr %15, align 1
  %index.next = add nuw i64 %index, %13
  %16 = icmp eq i64 %index.next, %n.vec
  br i1 %16, label %middle.block, label %vector.body

middle.block:                                     ; preds = %vector.body
  %cmp.n = icmp eq i64 %n.vec, %size
  br i1 %cmp.n, label %return, label %for.ph

for.ph:                                           ; preds = %vector.memcheck, %ph, %middle.block
  %i.ph = phi i64 [ 0, %vector.memcheck ], [ 0, %ph ], [ %n.vec, %middle.block ]
  br label %for.body

for.body:                                         ; preds = %for.ph, %for.body
  %i = phi i64 [ %inc, %for.body ], [ %i.ph, %for.ph ]
  %arrayidx = getelementptr inbounds i8, ptr %ptr, i64 %i
  %17 = load i8, ptr %arrayidx, align 1
  %arrayidx.st = getelementptr inbounds i8, ptr %0, i64 %i
  store i8 %17, ptr %arrayidx.st, align 1
  %inc = add nuw i64 %i, 1
  %exitcond.not = icmp eq i64 %inc, %size
  br i1 %exitcond.not, label %return, label %for.body

return:                                           ; preds = %for.body, %middle.block, %if.end, %exit, %entry
  %retval = phi ptr [ null, %entry ], [ null, %exit ], [ null, %if.end ], [ %0, %middle.block ], [ %0, %for.body ]
  ret ptr %retval
}

declare i64 @llvm.vscale.i64()
declare i64 @llvm.umax.i64(i64, i64)
