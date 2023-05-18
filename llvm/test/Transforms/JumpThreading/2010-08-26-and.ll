; RUN: opt -passes=jump-threading -S < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%class.StringSwitch = type { ptr, i32, i32, i8 }

@.str = private constant [4 x i8] c"red\00"       ; <ptr> [#uses=1]
@.str1 = private constant [7 x i8] c"orange\00"   ; <ptr> [#uses=1]
@.str2 = private constant [7 x i8] c"yellow\00"   ; <ptr> [#uses=1]
@.str3 = private constant [6 x i8] c"green\00"    ; <ptr> [#uses=1]
@.str4 = private constant [5 x i8] c"blue\00"     ; <ptr> [#uses=1]
@.str5 = private constant [7 x i8] c"indigo\00"   ; <ptr> [#uses=1]
@.str6 = private constant [7 x i8] c"violet\00"   ; <ptr> [#uses=1]
@.str7 = private constant [12 x i8] c"Color = %d\0A\00" ; <ptr> [#uses=1]

define i32 @main(i32 %argc, ptr nocapture %argv) nounwind ssp {
entry:
  %cmp142 = icmp sgt i32 %argc, 1                 ; <i1> [#uses=1]
  br i1 %cmp142, label %bb.nph, label %for.end

bb.nph:                                           ; preds = %entry
  %tmp = add i32 %argc, -2                        ; <i32> [#uses=1]
  %tmp144 = zext i32 %tmp to i64                  ; <i64> [#uses=1]
  %tmp145 = add i64 %tmp144, 1                    ; <i64> [#uses=1]
  br label %land.lhs.true.i

land.lhs.true.i:                                  ; preds = %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134, %bb.nph
  %retval.0.i.pre161 = phi i32 [ undef, %bb.nph ], [ %retval.0.i.pre, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134 ] ; <i32> [#uses=3]
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp146, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134 ] ; <i64> [#uses=1]
  %tmp146 = add i64 %indvar, 1                    ; <i64> [#uses=3]
  %arrayidx = getelementptr ptr, ptr %argv, i64 %tmp146 ; <ptr> [#uses=1]
  %tmp6 = load ptr, ptr %arrayidx, align 8            ; <ptr> [#uses=8]
  %call.i.i = call i64 @strlen(ptr %tmp6) nounwind ; <i64> [#uses=1]
  %conv.i.i = trunc i64 %call.i.i to i32          ; <i32> [#uses=6]\
; CHECK: switch i32 %conv.i.i
; CHECK-NOT: if.then.i40
; CHECK: }
  switch i32 %conv.i.i, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit [
    i32 3, label %land.lhs.true5.i
    i32 6, label %land.lhs.true5.i37
  ]

land.lhs.true5.i:                                 ; preds = %land.lhs.true.i
  %call.i = call i32 @memcmp(ptr @.str, ptr %tmp6, i64 4) nounwind ; <i32> [#uses=1]
  %cmp9.i = icmp eq i32 %call.i, 0                ; <i1> [#uses=1]
  br i1 %cmp9.i, label %_ZN12StringSwitchI5ColorE4CaseILj4EEERS1_RAT__KcRKS0_.exit, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit

_ZN12StringSwitchI5ColorE4CaseILj4EEERS1_RAT__KcRKS0_.exit: ; preds = %land.lhs.true5.i
  br label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit

land.lhs.true5.i37:                               ; preds = %land.lhs.true.i
  %call.i35 = call i32 @memcmp(ptr @.str1, ptr %tmp6, i64 7) nounwind ; <i32> [#uses=1]
  %cmp9.i36 = icmp eq i32 %call.i35, 0            ; <i1> [#uses=1]
  br i1 %cmp9.i36, label %if.then.i40, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit

if.then.i40:                                      ; preds = %land.lhs.true5.i37
  br label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit

_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit: ; preds = %if.then.i40, %land.lhs.true5.i37, %_ZN12StringSwitchI5ColorE4CaseILj4EEERS1_RAT__KcRKS0_.exit, %land.lhs.true5.i, %land.lhs.true.i
  %retval.0.i.pre159 = phi i32 [ 1, %_ZN12StringSwitchI5ColorE4CaseILj4EEERS1_RAT__KcRKS0_.exit ], [ %retval.0.i.pre161, %land.lhs.true5.i37 ], [ 2, %if.then.i40 ], [ %retval.0.i.pre161, %land.lhs.true5.i ], [ %retval.0.i.pre161, %land.lhs.true.i ] ; <i32> [#uses=2]
  %tmp2.i44 = phi i8 [ 1, %_ZN12StringSwitchI5ColorE4CaseILj4EEERS1_RAT__KcRKS0_.exit ], [ 0, %land.lhs.true5.i37 ], [ 1, %if.then.i40 ], [ 0, %land.lhs.true5.i ], [ 0, %land.lhs.true.i ] ; <i8> [#uses=3]
  %tobool.i46 = icmp eq i8 %tmp2.i44, 0           ; <i1> [#uses=1]
  %cmp.i49 = icmp eq i32 %conv.i.i, 6             ; <i1> [#uses=1]
  %or.cond = and i1 %tobool.i46, %cmp.i49         ; <i1> [#uses=1]
  br i1 %or.cond, label %land.lhs.true5.i55, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60

land.lhs.true5.i55:                               ; preds = %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit
  %call.i53 = call i32 @memcmp(ptr @.str2, ptr %tmp6, i64 7) nounwind ; <i32> [#uses=1]
  %cmp9.i54 = icmp eq i32 %call.i53, 0            ; <i1> [#uses=1]
  br i1 %cmp9.i54, label %if.then.i58, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60

if.then.i58:                                      ; preds = %land.lhs.true5.i55
  br label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60

_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60: ; preds = %if.then.i58, %land.lhs.true5.i55, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit
  %retval.0.i.pre158 = phi i32 [ %retval.0.i.pre159, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit ], [ %retval.0.i.pre159, %land.lhs.true5.i55 ], [ 3, %if.then.i58 ] ; <i32> [#uses=2]
  %tmp2.i63 = phi i8 [ %tmp2.i44, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit ], [ %tmp2.i44, %land.lhs.true5.i55 ], [ 1, %if.then.i58 ] ; <i8> [#uses=3]
  %tmp14.i64 = and i8 %tmp2.i63, 1                ; <i8> [#uses=1]
  %tobool.i65 = icmp eq i8 %tmp14.i64, 0          ; <i1> [#uses=1]
  %cmp.i68 = icmp eq i32 %conv.i.i, 5             ; <i1> [#uses=1]
  %or.cond168 = and i1 %tobool.i65, %cmp.i68      ; <i1> [#uses=1]
  br i1 %or.cond168, label %land.lhs.true5.i74, label %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit

land.lhs.true5.i74:                               ; preds = %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60
  %call.i72 = call i32 @memcmp(ptr @.str3, ptr %tmp6, i64 6) nounwind ; <i32> [#uses=1]
  %cmp9.i73 = icmp eq i32 %call.i72, 0            ; <i1> [#uses=1]
  br i1 %cmp9.i73, label %if.then.i77, label %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit

if.then.i77:                                      ; preds = %land.lhs.true5.i74
  br label %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit

_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit: ; preds = %if.then.i77, %land.lhs.true5.i74, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60
  %retval.0.i.pre157 = phi i32 [ %retval.0.i.pre158, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60 ], [ %retval.0.i.pre158, %land.lhs.true5.i74 ], [ 4, %if.then.i77 ] ; <i32> [#uses=2]
  %tmp2.i81 = phi i8 [ %tmp2.i63, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit60 ], [ %tmp2.i63, %land.lhs.true5.i74 ], [ 1, %if.then.i77 ] ; <i8> [#uses=3]
  %tmp14.i82 = and i8 %tmp2.i81, 1                ; <i8> [#uses=1]
  %tobool.i83 = icmp eq i8 %tmp14.i82, 0          ; <i1> [#uses=1]
  %cmp.i86 = icmp eq i32 %conv.i.i, 4             ; <i1> [#uses=1]
  %or.cond169 = and i1 %tobool.i83, %cmp.i86      ; <i1> [#uses=1]
  br i1 %or.cond169, label %land.lhs.true5.i92, label %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit

land.lhs.true5.i92:                               ; preds = %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit
  %call.i90 = call i32 @memcmp(ptr @.str4, ptr %tmp6, i64 5) nounwind ; <i32> [#uses=1]
  %cmp9.i91 = icmp eq i32 %call.i90, 0            ; <i1> [#uses=1]
  br i1 %cmp9.i91, label %if.then.i95, label %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit

if.then.i95:                                      ; preds = %land.lhs.true5.i92
  br label %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit

_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit: ; preds = %if.then.i95, %land.lhs.true5.i92, %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit
  %retval.0.i.pre156 = phi i32 [ %retval.0.i.pre157, %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit ], [ %retval.0.i.pre157, %land.lhs.true5.i92 ], [ 5, %if.then.i95 ] ; <i32> [#uses=2]
  %tmp2.i99 = phi i8 [ %tmp2.i81, %_ZN12StringSwitchI5ColorE4CaseILj6EEERS1_RAT__KcRKS0_.exit ], [ %tmp2.i81, %land.lhs.true5.i92 ], [ 1, %if.then.i95 ] ; <i8> [#uses=3]
  %tmp14.i100 = and i8 %tmp2.i99, 1               ; <i8> [#uses=1]
  %tobool.i101 = icmp eq i8 %tmp14.i100, 0        ; <i1> [#uses=1]
  %cmp.i104 = icmp eq i32 %conv.i.i, 6            ; <i1> [#uses=1]
  %or.cond170 = and i1 %tobool.i101, %cmp.i104    ; <i1> [#uses=1]
  br i1 %or.cond170, label %land.lhs.true5.i110, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115

land.lhs.true5.i110:                              ; preds = %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit
  %call.i108 = call i32 @memcmp(ptr @.str5, ptr %tmp6, i64 7) nounwind ; <i32> [#uses=1]
  %cmp9.i109 = icmp eq i32 %call.i108, 0          ; <i1> [#uses=1]
  br i1 %cmp9.i109, label %if.then.i113, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115

if.then.i113:                                     ; preds = %land.lhs.true5.i110
  br label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115

_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115: ; preds = %if.then.i113, %land.lhs.true5.i110, %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit
  %retval.0.i.pre155 = phi i32 [ %retval.0.i.pre156, %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit ], [ %retval.0.i.pre156, %land.lhs.true5.i110 ], [ 6, %if.then.i113 ] ; <i32> [#uses=2]
  %tmp2.i118 = phi i8 [ %tmp2.i99, %_ZN12StringSwitchI5ColorE4CaseILj5EEERS1_RAT__KcRKS0_.exit ], [ %tmp2.i99, %land.lhs.true5.i110 ], [ 1, %if.then.i113 ] ; <i8> [#uses=3]
  %tmp14.i119 = and i8 %tmp2.i118, 1              ; <i8> [#uses=1]
  %tobool.i120 = icmp eq i8 %tmp14.i119, 0        ; <i1> [#uses=1]
  %cmp.i123 = icmp eq i32 %conv.i.i, 6            ; <i1> [#uses=1]
  %or.cond171 = and i1 %tobool.i120, %cmp.i123    ; <i1> [#uses=1]
  br i1 %or.cond171, label %land.lhs.true5.i129, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134

land.lhs.true5.i129:                              ; preds = %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115
  %call.i127 = call i32 @memcmp(ptr @.str6, ptr %tmp6, i64 7) nounwind ; <i32> [#uses=1]
  %cmp9.i128 = icmp eq i32 %call.i127, 0          ; <i1> [#uses=1]
  br i1 %cmp9.i128, label %if.then.i132, label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134

if.then.i132:                                     ; preds = %land.lhs.true5.i129
  br label %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134

_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134: ; preds = %if.then.i132, %land.lhs.true5.i129, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115
  %retval.0.i.pre = phi i32 [ %retval.0.i.pre155, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115 ], [ %retval.0.i.pre155, %land.lhs.true5.i129 ], [ 7, %if.then.i132 ] ; <i32> [#uses=2]
  %tmp2.i137 = phi i8 [ %tmp2.i118, %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit115 ], [ %tmp2.i118, %land.lhs.true5.i129 ], [ 1, %if.then.i132 ] ; <i8> [#uses=1]
  %tmp7.i138 = and i8 %tmp2.i137, 1               ; <i8> [#uses=1]
  %tobool.i139 = icmp eq i8 %tmp7.i138, 0         ; <i1> [#uses=1]
  %retval.0.i = select i1 %tobool.i139, i32 0, i32 %retval.0.i.pre ; <i32> [#uses=1]
  %call22 = call i32 (ptr, ...) @printf(ptr @.str7, i32 %retval.0.i) ; <i32> [#uses=0]
  %exitcond = icmp eq i64 %tmp146, %tmp145        ; <i1> [#uses=1]
  br i1 %exitcond, label %for.end, label %land.lhs.true.i

for.end:                                          ; preds = %_ZN12StringSwitchI5ColorE4CaseILj7EEERS1_RAT__KcRKS0_.exit134, %entry
  ret i32 0
}

declare i32 @printf(ptr nocapture, ...) nounwind

declare i32 @memcmp(ptr nocapture, ptr nocapture, i64) nounwind readonly

declare i64 @strlen(ptr nocapture) nounwind readonly
