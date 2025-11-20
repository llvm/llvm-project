; The goal of this test case is to ensure that translation does not crash when during branching
; optimization analyzeBranch() requires helper methods of removeBranch() and insertBranch()
; to manage subsequent operations.

; RUN: llc -O0 -mtriple=spirv64-unknown-linux %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; RUN: llc -O0 -mtriple=spirv32-unknown-linux %s -o - | FileCheck %s
; TODO: %if spirv-tools %{ llc -O0 -mtriple=spirv32-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK: OpFunction

%struct = type { %arr }
%arr = type { [3 x i64] }

@.str.6 = private unnamed_addr addrspace(1) constant [3 x i8] c", \00", align 1
@.str.20 = private unnamed_addr addrspace(1) constant [6 x i8] c"item(\00", align 1
@.str.21 = private unnamed_addr addrspace(1) constant [8 x i8] c"range: \00", align 1
@.str.22 = private unnamed_addr addrspace(1) constant [7 x i8] c", id: \00", align 1

define spir_func i32 @foo(ptr addrspace(4) %Buf, ptr addrspace(4) %Item) {
entry:
  %ref.tmp = alloca %struct
  %ref.tmp7 = alloca %struct
  br label %for.cond.i

for.cond.i:                                       ; preds = %for.inc.i, %entry
  %Len.0.i = phi i32 [ 0, %entry ], [ %inc.i, %for.inc.i ]
  %idxprom.i = zext i32 %Len.0.i to i64
  %arrayidx.i = getelementptr inbounds i8, ptr addrspace(1) @.str.20, i64 %idxprom.i
  %0 = load i8, ptr addrspace(1) %arrayidx.i
  %cmp.not.i = icmp eq i8 %0, 0
  br i1 %cmp.not.i, label %for.cond1.i, label %for.inc.i

for.inc.i:                                        ; preds = %for.cond.i
  %inc.i = add i32 %Len.0.i, 1
  br label %for.cond.i, !llvm.loop !1

for.cond1.i:                                      ; preds = %for.body3.i, %for.cond.i
  %I.0.i = phi i32 [ %inc9.i, %for.body3.i ], [ 0, %for.cond.i ]
  %cmp2.i = icmp ult i32 %I.0.i, %Len.0.i
  br i1 %cmp2.i, label %for.body3.i, label %for.cond.preheader

for.cond.preheader:                               ; preds = %for.cond1.i
  %MIndex.i = getelementptr inbounds i8, ptr addrspace(4) %Item, i64 24
  br label %for.cond

for.body3.i:                                      ; preds = %for.cond1.i
  %idxprom4.i = zext i32 %I.0.i to i64
  %arrayidx5.i = getelementptr inbounds i8, ptr addrspace(1) @.str.20, i64 %idxprom4.i
  %1 = load i8, ptr addrspace(1) %arrayidx5.i
  %arrayidx7.i = getelementptr inbounds i8, ptr addrspace(4) %Buf, i64 %idxprom4.i
  store i8 %1, ptr addrspace(4) %arrayidx7.i
  %inc9.i = add nuw i32 %I.0.i, 1
  br label %for.cond1.i, !llvm.loop !2

for.cond:                                         ; preds = %exit, %for.cond.preheader
  %Len.0 = phi i32 [ %add9, %exit ], [ %Len.0.i, %for.cond.preheader ]
  %I.0 = phi i32 [ %inc, %exit ], [ 0, %for.cond.preheader ]
  %cmp = icmp ult i32 %I.0, 2
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.cond
  %inc10 = add i32 %Len.0, 1
  %idxprom = zext i32 %Len.0 to i64
  %arrayidx = getelementptr inbounds i8, ptr addrspace(4) %Buf, i64 %idxprom
  store i8 41, ptr addrspace(4) %arrayidx
  ret i32 %inc10

for.body:                                         ; preds = %for.cond
  %idx.ext = zext i32 %Len.0 to i64
  %add.ptr = getelementptr inbounds i8, ptr addrspace(4) %Buf, i64 %idx.ext
  %cmp1 = icmp eq i32 %I.0, 0
  %cond = select i1 %cmp1, ptr addrspace(1) @.str.21, ptr addrspace(1) @.str.22
  br label %for.cond.i25

for.cond.i25:                                     ; preds = %for.inc.i30, %for.body
  %Len.0.i26 = phi i32 [ 0, %for.body ], [ %inc.i31, %for.inc.i30 ]
  %idxprom.i27 = zext i32 %Len.0.i26 to i64
  %arrayidx.i28 = getelementptr inbounds i8, ptr addrspace(1) %cond, i64 %idxprom.i27
  %2 = load i8, ptr addrspace(1) %arrayidx.i28
  %cmp.not.i29 = icmp eq i8 %2, 0
  br i1 %cmp.not.i29, label %for.cond1.i33, label %for.inc.i30

for.inc.i30:                                      ; preds = %for.cond.i25
  %inc.i31 = add i32 %Len.0.i26, 1
  br label %for.cond.i25, !llvm.loop !1

for.cond1.i33:                                    ; preds = %for.body3.i36, %for.cond.i25
  %I.0.i34 = phi i32 [ %inc9.i40, %for.body3.i36 ], [ 0, %for.cond.i25 ]
  %cmp2.i35 = icmp ult i32 %I.0.i34, %Len.0.i26
  br i1 %cmp2.i35, label %for.body3.i36, label %detail.exit

for.body3.i36:                                    ; preds = %for.cond1.i33
  %idxprom4.i37 = zext i32 %I.0.i34 to i64
  %arrayidx5.i38 = getelementptr inbounds i8, ptr addrspace(1) %cond, i64 %idxprom4.i37
  %3 = load i8, ptr addrspace(1) %arrayidx5.i38
  %arrayidx7.i39 = getelementptr inbounds i8, ptr addrspace(4) %add.ptr, i64 %idxprom4.i37
  store i8 %3, ptr addrspace(4) %arrayidx7.i39
  %inc9.i40 = add nuw i32 %I.0.i34, 1
  br label %for.cond1.i33, !llvm.loop !2

detail.exit:          ; preds = %for.cond1.i33
  %add3 = add i32 %Len.0, %Len.0.i26
  %idx.ext4 = zext i32 %add3 to i64
  %add.ptr5 = getelementptr inbounds i8, ptr addrspace(4) %Buf, i64 %idx.ext4
  br i1 %cmp1, label %cond.true, label %cond.false

cond.true:                                        ; preds = %detail.exit
  call void @llvm.memcpy.p0.p4.i64(ptr align 8 %ref.tmp7, ptr addrspace(4) align 8 %Item, i64 24, i1 false)
  call void @llvm.memset.p0.i64(ptr align 8 %ref.tmp, i8 0, i64 24, i1 false)
  br label %for.cond.i42

for.cond.i42:                                     ; preds = %for.body.i, %cond.true
  %i.0.i = phi i32 [ 0, %cond.true ], [ %inc.i45, %for.body.i ]
  %cmp.i = icmp ult i32 %i.0.i, 3
  br i1 %cmp.i, label %for.body.i, label %cond.end

for.body.i:                                       ; preds = %for.cond.i42
  %idxprom.i43 = zext nneg i32 %i.0.i to i64
  %arrayidx.i44 = getelementptr inbounds [3 x i64], ptr %ref.tmp7, i64 0, i64 %idxprom.i43
  %4 = load i64, ptr %arrayidx.i44, align 8
  %arrayidx.i.i = getelementptr inbounds [3 x i64], ptr %ref.tmp, i64 0, i64 %idxprom.i43
  store i64 %4, ptr %arrayidx.i.i, align 8
  %inc.i45 = add nuw nsw i32 %i.0.i, 1
  br label %for.cond.i42, !llvm.loop !3

cond.false:                                       ; preds = %detail.exit
  call void @llvm.memcpy.p0.p4.i64(ptr align 8 %ref.tmp, ptr addrspace(4) align 8 %MIndex.i, i64 24, i1 false)
  br label %cond.end

cond.end:                                         ; preds = %cond.false, %for.cond.i42
  store i8 123, ptr addrspace(4) %add.ptr5
  br label %for.cond.i46

for.cond.i46:                                     ; preds = %for.inc.i52, %cond.end
  %Len.0.i47 = phi i32 [ 1, %cond.end ], [ %Len.1.i, %for.inc.i52 ]
  %I.0.i48 = phi i32 [ 0, %cond.end ], [ %inc7.i, %for.inc.i52 ]
  %cmp.i49 = icmp ult i32 %I.0.i48, 3
  br i1 %cmp.i49, label %for.body.i50, label %exit

for.body.i50:                                     ; preds = %for.cond.i46
  %idxprom.i.i = zext nneg i32 %I.0.i48 to i64
  %arrayidx.i.i51 = getelementptr inbounds [3 x i64], ptr %ref.tmp, i64 0, i64 %idxprom.i.i
  %5 = load i64, ptr %arrayidx.i.i51, align 8
  %idx.ext.i = zext i32 %Len.0.i47 to i64
  %add.ptr.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr5, i64 %idx.ext.i
  br label %do.body.i.i.i

do.body.i.i.i:                                    ; preds = %do.body.i.i.i, %for.body.i50
  %Val.addr.0.i.i.i = phi i64 [ %5, %for.body.i50 ], [ %div.i.i.i, %do.body.i.i.i ]
  %NumDigits.0.i.i.i = phi i32 [ 0, %for.body.i50 ], [ %inc.i.i.i, %do.body.i.i.i ]
  %Val.addr.0.i.i.i.frozen = freeze i64 %Val.addr.0.i.i.i
  %div.i.i.i = udiv i64 %Val.addr.0.i.i.i.frozen, 10
  %6 = mul i64 %div.i.i.i, 10
  %rem.i.i.i.decomposed = sub i64 %Val.addr.0.i.i.i.frozen, %6
  %7 = trunc i64 %rem.i.i.i.decomposed to i8
  %retval.0.i.i.i.i = or disjoint i8 %7, 48
  %inc.i.i.i = add i32 %NumDigits.0.i.i.i, 1
  %idxprom.i.i.i = zext i32 %NumDigits.0.i.i.i to i64
  %arrayidx.i.i.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr.i, i64 %idxprom.i.i.i
  store i8 %retval.0.i.i.i.i, ptr addrspace(4) %arrayidx.i.i.i
  %tobool.not.i.i.i = icmp ult i64 %Val.addr.0.i.i.i, 10
  br i1 %tobool.not.i.i.i, label %while.cond.i.i.i, label %do.body.i.i.i, !llvm.loop !4

while.cond.i.i.i:                                 ; preds = %while.body.i.i.i, %do.body.i.i.i
  %J.0.i.i.i = phi i32 [ %inc.i54.i.i, %while.body.i.i.i ], [ 0, %do.body.i.i.i ]
  %I.0.in.i.i.i = phi i32 [ %I.0.i.i.i, %while.body.i.i.i ], [ %inc.i.i.i, %do.body.i.i.i ]
  %I.0.i.i.i = add i32 %I.0.in.i.i.i, -1
  %cmp.i.i.i = icmp sgt i32 %I.0.i.i.i, %J.0.i.i.i
  br i1 %cmp.i.i.i, label %while.body.i.i.i, label %enable.exit

while.body.i.i.i:                                 ; preds = %while.cond.i.i.i
  %idxprom.i52.i.i = sext i32 %I.0.i.i.i to i64
  %arrayidx.i53.i.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr.i, i64 %idxprom.i52.i.i
  %8 = load i8, ptr addrspace(4) %arrayidx.i53.i.i
  %idxprom1.i.i.i = zext nneg i32 %J.0.i.i.i to i64
  %arrayidx2.i.i.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr.i, i64 %idxprom1.i.i.i
  %9 = load i8, ptr addrspace(4) %arrayidx2.i.i.i
  store i8 %9, ptr addrspace(4) %arrayidx.i53.i.i
  store i8 %8, ptr addrspace(4) %arrayidx2.i.i.i
  %inc.i54.i.i = add nuw nsw i32 %J.0.i.i.i, 1
  br label %while.cond.i.i.i, !llvm.loop !5

enable.exit: ; preds = %while.cond.i.i.i
  %add.i = add i32 %Len.0.i47, %inc.i.i.i
  %cmp2.not.i = icmp eq i32 %I.0.i48, 2
  br i1 %cmp2.not.i, label %for.inc.i52, label %if.then.i

if.then.i:                                        ; preds = %enable.exit
  %idx.ext3.i = zext i32 %add.i to i64
  %add.ptr4.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr5, i64 %idx.ext3.i
  br label %for.cond.i.i

for.cond.i.i:                                     ; preds = %for.inc.i.i, %if.then.i
  %Len.0.i.i = phi i32 [ 0, %if.then.i ], [ %inc.i.i, %for.inc.i.i ]
  %idxprom.i24.i = zext i32 %Len.0.i.i to i64
  %arrayidx.i25.i = getelementptr inbounds i8, ptr addrspace(1) @.str.6, i64 %idxprom.i24.i
  %10 = load i8, ptr addrspace(1) %arrayidx.i25.i
  %cmp.not.i.i = icmp eq i8 %10, 0
  br i1 %cmp.not.i.i, label %for.cond1.i.i, label %for.inc.i.i

for.inc.i.i:                                      ; preds = %for.cond.i.i
  %inc.i.i = add i32 %Len.0.i.i, 1
  br label %for.cond.i.i, !llvm.loop !1

for.cond1.i.i:                                    ; preds = %for.body3.i.i, %for.cond.i.i
  %I.0.i.i = phi i32 [ %inc9.i.i, %for.body3.i.i ], [ 0, %for.cond.i.i ]
  %cmp2.i.i = icmp ult i32 %I.0.i.i, %Len.0.i.i
  br i1 %cmp2.i.i, label %for.body3.i.i, label %append.exit

for.body3.i.i:                                    ; preds = %for.cond1.i.i
  %idxprom4.i.i = zext i32 %I.0.i.i to i64
  %arrayidx5.i.i = getelementptr inbounds i8, ptr addrspace(1) @.str.6, i64 %idxprom4.i.i
  %11 = load i8, ptr addrspace(1) %arrayidx5.i.i
  %arrayidx7.i.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr4.i, i64 %idxprom4.i.i
  store i8 %11, ptr addrspace(4) %arrayidx7.i.i
  %inc9.i.i = add nuw i32 %I.0.i.i, 1
  br label %for.cond1.i.i, !llvm.loop !2

append.exit:          ; preds = %for.cond1.i.i
  %add6.i = add i32 %add.i, %Len.0.i.i
  br label %for.inc.i52

for.inc.i52:                                      ; preds = %append.exit, %enable.exit
  %Len.1.i = phi i32 [ %add6.i, %append.exit ], [ %add.i, %enable.exit ]
  %inc7.i = add nuw nsw i32 %I.0.i48, 1
  br label %for.cond.i46, !llvm.loop !6

exit: ; preds = %for.cond.i46
  %inc8.i = add i32 %Len.0.i47, 1
  %idxprom9.i = zext i32 %Len.0.i47 to i64
  %arrayidx10.i = getelementptr inbounds i8, ptr addrspace(4) %add.ptr5, i64 %idxprom9.i
  store i8 125, ptr addrspace(4) %arrayidx10.i
  %add9 = add i32 %add3, %inc8.i
  %inc = add nuw nsw i32 %I.0, 1
  br label %for.cond, !llvm.loop !7
}

declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg)
declare void @llvm.memcpy.p0.p4.i64(ptr noalias nocapture writeonly, ptr addrspace(4) noalias nocapture readonly, i64, i1 immarg)

!0 = !{!"llvm.loop.mustprogress"}
!1 = distinct !{!1, !0}
!2 = distinct !{!2, !0}
!3 = distinct !{!3, !0}
!4 = distinct !{!4, !0}
!5 = distinct !{!5, !0}
!6 = distinct !{!6, !0}
!7 = distinct !{!7, !0}
