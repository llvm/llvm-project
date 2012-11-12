; RUN: llc < %s -O3
; PR4626
; ModuleID = '<stdin>'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:32:32"
target triple = "i386-pc-linux-gnu"
@g_3 = common global i8 0, align 1		; <i8*> [#uses=2]

define signext i8 @safe_mul_func_int16_t_s_s(i32 %_si1, i8 signext %_si2) nounwind readnone {
entry:
	%tobool = icmp eq i32 %_si1, 0		; <i1> [#uses=1]
	%cmp = icmp sgt i8 %_si2, 0		; <i1> [#uses=2]
	%or.cond = or i1 %cmp, %tobool		; <i1> [#uses=1]
	br i1 %or.cond, label %lor.rhs, label %land.lhs.true3

land.lhs.true3:		; preds = %entry
	%conv5 = sext i8 %_si2 to i32		; <i32> [#uses=1]
	%cmp7 = icmp slt i32 %conv5, %_si1		; <i1> [#uses=1]
	br i1 %cmp7, label %cond.end, label %lor.rhs

lor.rhs:		; preds = %land.lhs.true3, %entry
	%cmp10.not = icmp slt i32 %_si1, 1		; <i1> [#uses=1]
	%or.cond23 = and i1 %cmp, %cmp10.not		; <i1> [#uses=1]
	br i1 %or.cond23, label %lor.end, label %cond.false

lor.end:		; preds = %lor.rhs
	%tobool19 = icmp ne i8 %_si2, 0		; <i1> [#uses=2]
	%lor.ext = zext i1 %tobool19 to i32		; <i32> [#uses=1]
	br i1 %tobool19, label %cond.end, label %cond.false

cond.false:		; preds = %lor.end, %lor.rhs
	%conv21 = sext i8 %_si2 to i32		; <i32> [#uses=1]
	br label %cond.end

cond.end:		; preds = %cond.false, %lor.end, %land.lhs.true3
	%cond = phi i32 [ %conv21, %cond.false ], [ 1, %land.lhs.true3 ], [ %lor.ext, %lor.end ]		; <i32> [#uses=1]
	%conv22 = trunc i32 %cond to i8		; <i8> [#uses=1]
	ret i8 %conv22
}

define i32 @func_34(i8 signext %p_35) nounwind readonly {
entry:
	%tobool = icmp eq i8 %p_35, 0		; <i1> [#uses=1]
	br i1 %tobool, label %lor.lhs.false, label %if.then

lor.lhs.false:		; preds = %entry
	%tmp1 = load i8* @g_3		; <i8> [#uses=1]
	%tobool3 = icmp eq i8 %tmp1, 0		; <i1> [#uses=1]
	br i1 %tobool3, label %return, label %if.then

if.then:		; preds = %lor.lhs.false, %entry
	%tmp4 = load i8* @g_3		; <i8> [#uses=1]
	%conv5 = sext i8 %tmp4 to i32		; <i32> [#uses=1]
	ret i32 %conv5

return:		; preds = %lor.lhs.false
	ret i32 0
}

define void @foo(i32 %p_5) noreturn nounwind {
entry:
	%cmp = icmp sgt i32 %p_5, 0		; <i1> [#uses=2]
	%call = tail call i32 @safe() nounwind		; <i32> [#uses=1]
	%conv1 = trunc i32 %call to i8		; <i8> [#uses=3]
	%tobool.i = xor i1 %cmp, true		; <i1> [#uses=3]
	%cmp.i = icmp sgt i8 %conv1, 0		; <i1> [#uses=3]
	%or.cond.i = or i1 %cmp.i, %tobool.i		; <i1> [#uses=1]
	br i1 %or.cond.i, label %lor.rhs.i, label %land.lhs.true3.i

land.lhs.true3.i:		; preds = %entry
	%xor = zext i1 %cmp to i32		; <i32> [#uses=1]
	%conv5.i = sext i8 %conv1 to i32		; <i32> [#uses=1]
	%cmp7.i = icmp slt i32 %conv5.i, %xor		; <i1> [#uses=1]
	%cmp7.i.not = xor i1 %cmp7.i, true		; <i1> [#uses=1]
	%or.cond23.i = and i1 %cmp.i, %tobool.i		; <i1> [#uses=1]
	%or.cond = and i1 %cmp7.i.not, %or.cond23.i		; <i1> [#uses=1]
	br i1 %or.cond, label %lor.end.i, label %for.inc

lor.rhs.i:		; preds = %entry
	%or.cond23.i.old = and i1 %cmp.i, %tobool.i		; <i1> [#uses=1]
	br i1 %or.cond23.i.old, label %lor.end.i, label %for.inc

lor.end.i:		; preds = %lor.rhs.i, %land.lhs.true3.i
	%tobool19.i = icmp eq i8 %conv1, 0		; <i1> [#uses=0]
	br label %for.inc

for.inc:		; preds = %for.inc, %lor.end.i, %lor.rhs.i, %land.lhs.true3.i
	br label %for.inc
}

define i32 @func_35(i8 signext %p_35) nounwind readonly {
entry:
  %tobool = icmp eq i8 %p_35, 0                   ; <i1> [#uses=1]
  br i1 %tobool, label %lor.lhs.false, label %if.then

lor.lhs.false:                                    ; preds = %entry
  %tmp1 = load i8* @g_3                           ; <i8> [#uses=1]
  %tobool3 = icmp eq i8 %tmp1, 0                  ; <i1> [#uses=1]
  br i1 %tobool3, label %return, label %if.then

if.then:                                          ; preds = %lor.lhs.false, %entry
  %tmp4 = load i8* @g_3                           ; <i8> [#uses=1]
  %conv5 = sext i8 %tmp4 to i32                   ; <i32> [#uses=1]
  ret i32 %conv5

return:                                           ; preds = %lor.lhs.false
  ret i32 0
}

define void @bar(i32 %p_5) noreturn nounwind {
entry:
  %cmp = icmp sgt i32 %p_5, 0                     ; <i1> [#uses=2]
  %call = tail call i32 @safe() nounwind          ; <i32> [#uses=1]
  %conv1 = trunc i32 %call to i8                  ; <i8> [#uses=3]
  %tobool.i = xor i1 %cmp, true                   ; <i1> [#uses=3]
  %cmp.i = icmp sgt i8 %conv1, 0                  ; <i1> [#uses=3]
  %or.cond.i = or i1 %cmp.i, %tobool.i            ; <i1> [#uses=1]
  br i1 %or.cond.i, label %lor.rhs.i, label %land.lhs.true3.i

land.lhs.true3.i:                                 ; preds = %entry
  %xor = zext i1 %cmp to i32                      ; <i32> [#uses=1]
  %conv5.i = sext i8 %conv1 to i32                ; <i32> [#uses=1]
  %cmp7.i = icmp slt i32 %conv5.i, %xor           ; <i1> [#uses=1]
  %cmp7.i.not = xor i1 %cmp7.i, true              ; <i1> [#uses=1]
  %or.cond23.i = and i1 %cmp.i, %tobool.i         ; <i1> [#uses=1]
  %or.cond = and i1 %cmp7.i.not, %or.cond23.i     ; <i1> [#uses=1]
  br i1 %or.cond, label %lor.end.i, label %for.inc

lor.rhs.i:                                        ; preds = %entry
  %or.cond23.i.old = and i1 %cmp.i, %tobool.i     ; <i1> [#uses=1]
  br i1 %or.cond23.i.old, label %lor.end.i, label %for.inc

lor.end.i:                                        ; preds = %lor.rhs.i, %land.lhs.true3.i
  %tobool19.i = icmp eq i8 %conv1, 0              ; <i1> [#uses=0]
  br label %for.inc

for.inc:                                          ; preds = %for.inc, %lor.end.i, %lor.rhs.i, %land.lhs.true3.i
  br label %for.inc
}

declare i32 @safe()
