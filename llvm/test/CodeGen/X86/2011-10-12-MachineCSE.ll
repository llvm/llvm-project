; RUN: llc -verify-machineinstrs < %s
; <rdar://problem/10270968>
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.2"

%struct.optab = type { i32, [59 x %struct.anon.3] }
%struct.anon.3 = type { i32, ptr }
%struct.rtx_def = type { [2 x i8], i8, i8, [1 x %union.rtunion_def] }
%union.rtunion_def = type { i64 }
%struct.insn_data = type { ptr, ptr, ptr, ptr, i8, i8, i8, i8 }
%struct.insn_operand_data = type { ptr, ptr, [2 x i8], i8, i8 }

@optab_table = external global [49 x ptr], align 16
@insn_data = external constant [0 x %struct.insn_data]

define ptr @gen_add3_insn(ptr %r0, ptr %r1, ptr %c) nounwind uwtable ssp {
entry:
  %0 = load i32, ptr %r0, align 8
  %1 = lshr i32 %0, 16
  %bf.clear = and i32 %1, 255
  %idxprom = sext i32 %bf.clear to i64
  %2 = load ptr, ptr @optab_table, align 8
  %handlers = getelementptr inbounds %struct.optab, ptr %2, i32 0, i32 1
  %arrayidx = getelementptr inbounds [59 x %struct.anon.3], ptr %handlers, i32 0, i64 %idxprom
  %3 = load i32, ptr %arrayidx, align 4
  %cmp = icmp eq i32 %3, 1317
  br i1 %cmp, label %if.then, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %idxprom1 = sext i32 %3 to i64
  %arrayidx2 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom1
  %operand = getelementptr inbounds %struct.insn_data, ptr %arrayidx2, i32 0, i32 3
  %4 = load ptr, ptr %operand, align 8
  %5 = load ptr, ptr %4, align 8
  %idxprom4 = sext i32 %3 to i64
  %arrayidx5 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom4
  %operand6 = getelementptr inbounds %struct.insn_data, ptr %arrayidx5, i32 0, i32 3
  %6 = load ptr, ptr %operand6, align 8
  %bf.field.offs = getelementptr i8, ptr %6, i32 16
  %7 = load i32, ptr %bf.field.offs, align 8
  %bf.clear8 = and i32 %7, 65535
  %call = tail call i32 %5(ptr %r0, i32 %bf.clear8)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %lor.lhs.false9, label %if.then

lor.lhs.false9:                                   ; preds = %lor.lhs.false
  %idxprom10 = sext i32 %3 to i64
  %arrayidx11 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom10
  %operand12 = getelementptr inbounds %struct.insn_data, ptr %arrayidx11, i32 0, i32 3
  %8 = load ptr, ptr %operand12, align 8
  %arrayidx13 = getelementptr inbounds %struct.insn_operand_data, ptr %8, i64 1
  %9 = load ptr, ptr %arrayidx13, align 8
  %idxprom15 = sext i32 %3 to i64
  %arrayidx16 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom15
  %operand17 = getelementptr inbounds %struct.insn_data, ptr %arrayidx16, i32 0, i32 3
  %10 = load ptr, ptr %operand17, align 8
  %arrayidx18 = getelementptr inbounds %struct.insn_operand_data, ptr %10, i64 1
  %bf.field.offs19 = getelementptr i8, ptr %arrayidx18, i32 16
  %11 = load i32, ptr %bf.field.offs19, align 8
  %bf.clear20 = and i32 %11, 65535
  %call21 = tail call i32 %9(ptr %r1, i32 %bf.clear20)
  %tobool22 = icmp ne i32 %call21, 0
  br i1 %tobool22, label %lor.lhs.false23, label %if.then

lor.lhs.false23:                                  ; preds = %lor.lhs.false9
  %idxprom24 = sext i32 %3 to i64
  %arrayidx25 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom24
  %operand26 = getelementptr inbounds %struct.insn_data, ptr %arrayidx25, i32 0, i32 3
  %12 = load ptr, ptr %operand26, align 8
  %arrayidx27 = getelementptr inbounds %struct.insn_operand_data, ptr %12, i64 2
  %13 = load ptr, ptr %arrayidx27, align 8
  %idxprom29 = sext i32 %3 to i64
  %arrayidx30 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom29
  %operand31 = getelementptr inbounds %struct.insn_data, ptr %arrayidx30, i32 0, i32 3
  %14 = load ptr, ptr %operand31, align 8
  %arrayidx32 = getelementptr inbounds %struct.insn_operand_data, ptr %14, i64 2
  %bf.field.offs33 = getelementptr i8, ptr %arrayidx32, i32 16
  %15 = load i32, ptr %bf.field.offs33, align 8
  %bf.clear34 = and i32 %15, 65535
  %call35 = tail call i32 %13(ptr %c, i32 %bf.clear34)
  %tobool36 = icmp ne i32 %call35, 0
  br i1 %tobool36, label %if.end, label %if.then

if.then:                                          ; preds = %lor.lhs.false23, %lor.lhs.false9, %lor.lhs.false, %entry
  br label %return

if.end:                                           ; preds = %lor.lhs.false23
  %idxprom37 = sext i32 %3 to i64
  %arrayidx38 = getelementptr inbounds [0 x %struct.insn_data], ptr @insn_data, i32 0, i64 %idxprom37
  %genfun = getelementptr inbounds %struct.insn_data, ptr %arrayidx38, i32 0, i32 2
  %16 = load ptr, ptr %genfun, align 8
  %call39 = tail call ptr (ptr, ...) %16(ptr %r0, ptr %r1, ptr %c)
  br label %return

return:                                           ; preds = %if.end, %if.then
  %17 = phi ptr [ %call39, %if.end ], [ null, %if.then ]
  ret ptr %17
}
