; RUN: opt %loadNPMPolly -polly-allow-nonaffine-loops '-passes=print<polly-function-scops>' -disable-output < %s 2>&1 | FileCheck %s
;
; CHECK:      Domain :=
; CHECK-NEXT:   { Stmt_while_cond_i__TO__while_end_i[] };
;
define i32 @func(i32 %param0, i32 %param1, ptr %param2) #3 {

entry:
  %var0 = alloca i32
  %var1 = alloca i32
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i.backedge, %entry
  %var2 = phi i32 [ %param0, %entry ], [ %var3, %while.cond.i.backedge ]
  %var3 = add nsw i32 %var2, 1
  %var4 = icmp slt i32 %var2, -1
  br i1 %var4, label %while.cond.i.backedge, label %if.end.i1.i

if.end.i1.i:                                    ; preds = %while.cond.i
  %var5 = sdiv i32 %var3, 64
  %var6 = icmp sgt i32 %param1, %var5
  br i1 %var6, label %exit1.i, label %while.cond.i.backedge

exit1.i:                          ; preds = %if.end.i1.i
  %var7 = srem i32 %var3, 64
  %var8 = sext i32 %var5 to i64
  %var9 = getelementptr inbounds i64, ptr %param2, i64 %var8
  %var10 = load i64, ptr %var9, align 8
  %var11 = zext i32 %var7 to i64
  %var12 = shl i64 1, %var11
  %var13 = and i64 %var10, %var12
  %var14 = icmp eq i64 %var13, 0
  store i32 %var2, ptr %var1
  store i32 %var3, ptr %var0
  br i1 %var14, label %while.cond.i.backedge, label %while.end.i

while.cond.i.backedge:                            ; preds = %exit1.i, %while.cond.i, %if.end.i1.i
  br label %while.cond.i

while.end.i:
  %var15 = load i32, ptr %var0
  %var16 = load i32, ptr %var1
  %var17 = add i32 %var15, %var16
  ret i32 %var17
}
