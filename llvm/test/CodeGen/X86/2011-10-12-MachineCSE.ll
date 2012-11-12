; RUN: llc -verify-machineinstrs < %s
; <rdar://problem/10270968>
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

%struct.optab = type { i32, [59 x %struct.anon.3] }
%struct.anon.3 = type { i32, %struct.rtx_def* }
%struct.rtx_def = type { [2 x i8], i8, i8, [1 x %union.rtunion_def] }
%union.rtunion_def = type { i64 }
%struct.insn_data = type { i8*, i8*, %struct.rtx_def* (%struct.rtx_def*, ...)*, %struct.insn_operand_data*, i8, i8, i8, i8 }
%struct.insn_operand_data = type { i32 (%struct.rtx_def*, i32)*, i8*, [2 x i8], i8, i8 }

@optab_table = external global [49 x %struct.optab*], align 16
@insn_data = external constant [0 x %struct.insn_data]

define %struct.rtx_def* @gen_add3_insn(%struct.rtx_def* %r0, %struct.rtx_def* %r1, %struct.rtx_def* %c) nounwind uwtable ssp {
entry:
  %0 = bitcast %struct.rtx_def* %r0 to i32*
  %1 = load i32* %0, align 8
  %2 = lshr i32 %1, 16
  %bf.clear = and i32 %2, 255
  %idxprom = sext i32 %bf.clear to i64
  %3 = load %struct.optab** getelementptr inbounds ([49 x %struct.optab*]* @optab_table, i32 0, i64 0), align 8, !tbaa !0
  %handlers = getelementptr inbounds %struct.optab* %3, i32 0, i32 1
  %arrayidx = getelementptr inbounds [59 x %struct.anon.3]* %handlers, i32 0, i64 %idxprom
  %insn_code = getelementptr inbounds %struct.anon.3* %arrayidx, i32 0, i32 0
  %4 = load i32* %insn_code, align 4, !tbaa !3
  %cmp = icmp eq i32 %4, 1317
  br i1 %cmp, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %idxprom1 = sext i32 %4 to i64
  %arrayidx2 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom1
  %operand = getelementptr inbounds %struct.insn_data* %arrayidx2, i32 0, i32 3
  %5 = load %struct.insn_operand_data** %operand, align 8, !tbaa !0
  %arrayidx3 = getelementptr inbounds %struct.insn_operand_data* %5, i64 0
  %predicate = getelementptr inbounds %struct.insn_operand_data* %arrayidx3, i32 0, i32 0
  %6 = load i32 (%struct.rtx_def*, i32)** %predicate, align 8, !tbaa !0
  %idxprom4 = sext i32 %4 to i64
  %arrayidx5 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom4
  %operand6 = getelementptr inbounds %struct.insn_data* %arrayidx5, i32 0, i32 3
  %7 = load %struct.insn_operand_data** %operand6, align 8, !tbaa !0
  %arrayidx7 = getelementptr inbounds %struct.insn_operand_data* %7, i64 0
  %8 = bitcast %struct.insn_operand_data* %arrayidx7 to i8*
  %bf.field.offs = getelementptr i8* %8, i32 16
  %9 = bitcast i8* %bf.field.offs to i32*
  %10 = load i32* %9, align 8
  %bf.clear8 = and i32 %10, 65535
  %call = tail call i32 %6(%struct.rtx_def* %r0, i32 %bf.clear8)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %lor.lhs.false9, label %if.then

lor.lhs.false9:                                   ; preds = %lor.lhs.false
  %idxprom10 = sext i32 %4 to i64
  %arrayidx11 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom10
  %operand12 = getelementptr inbounds %struct.insn_data* %arrayidx11, i32 0, i32 3
  %11 = load %struct.insn_operand_data** %operand12, align 8, !tbaa !0
  %arrayidx13 = getelementptr inbounds %struct.insn_operand_data* %11, i64 1
  %predicate14 = getelementptr inbounds %struct.insn_operand_data* %arrayidx13, i32 0, i32 0
  %12 = load i32 (%struct.rtx_def*, i32)** %predicate14, align 8, !tbaa !0
  %idxprom15 = sext i32 %4 to i64
  %arrayidx16 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom15
  %operand17 = getelementptr inbounds %struct.insn_data* %arrayidx16, i32 0, i32 3
  %13 = load %struct.insn_operand_data** %operand17, align 8, !tbaa !0
  %arrayidx18 = getelementptr inbounds %struct.insn_operand_data* %13, i64 1
  %14 = bitcast %struct.insn_operand_data* %arrayidx18 to i8*
  %bf.field.offs19 = getelementptr i8* %14, i32 16
  %15 = bitcast i8* %bf.field.offs19 to i32*
  %16 = load i32* %15, align 8
  %bf.clear20 = and i32 %16, 65535
  %call21 = tail call i32 %12(%struct.rtx_def* %r1, i32 %bf.clear20)
  %tobool22 = icmp ne i32 %call21, 0
  br i1 %tobool22, label %lor.lhs.false23, label %if.then

lor.lhs.false23:                                  ; preds = %lor.lhs.false9
  %idxprom24 = sext i32 %4 to i64
  %arrayidx25 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom24
  %operand26 = getelementptr inbounds %struct.insn_data* %arrayidx25, i32 0, i32 3
  %17 = load %struct.insn_operand_data** %operand26, align 8, !tbaa !0
  %arrayidx27 = getelementptr inbounds %struct.insn_operand_data* %17, i64 2
  %predicate28 = getelementptr inbounds %struct.insn_operand_data* %arrayidx27, i32 0, i32 0
  %18 = load i32 (%struct.rtx_def*, i32)** %predicate28, align 8, !tbaa !0
  %idxprom29 = sext i32 %4 to i64
  %arrayidx30 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom29
  %operand31 = getelementptr inbounds %struct.insn_data* %arrayidx30, i32 0, i32 3
  %19 = load %struct.insn_operand_data** %operand31, align 8, !tbaa !0
  %arrayidx32 = getelementptr inbounds %struct.insn_operand_data* %19, i64 2
  %20 = bitcast %struct.insn_operand_data* %arrayidx32 to i8*
  %bf.field.offs33 = getelementptr i8* %20, i32 16
  %21 = bitcast i8* %bf.field.offs33 to i32*
  %22 = load i32* %21, align 8
  %bf.clear34 = and i32 %22, 65535
  %call35 = tail call i32 %18(%struct.rtx_def* %c, i32 %bf.clear34)
  %tobool36 = icmp ne i32 %call35, 0
  br i1 %tobool36, label %if.end, label %if.then

if.then:                                          ; preds = %lor.lhs.false23, %lor.lhs.false9, %lor.lhs.false, %entry
  br label %return

if.end:                                           ; preds = %lor.lhs.false23
  %idxprom37 = sext i32 %4 to i64
  %arrayidx38 = getelementptr inbounds [0 x %struct.insn_data]* @insn_data, i32 0, i64 %idxprom37
  %genfun = getelementptr inbounds %struct.insn_data* %arrayidx38, i32 0, i32 2
  %23 = load %struct.rtx_def* (%struct.rtx_def*, ...)** %genfun, align 8, !tbaa !0
  %call39 = tail call %struct.rtx_def* (%struct.rtx_def*, ...)* %23(%struct.rtx_def* %r0, %struct.rtx_def* %r1, %struct.rtx_def* %c)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %24 = phi %struct.rtx_def* [ %call39, %if.end ], [ null, %if.then ]
  ret %struct.rtx_def* %24
}

!0 = metadata !{metadata !"any pointer", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA", null}
!3 = metadata !{metadata !"_ZTS9insn_code", metadata !1}
