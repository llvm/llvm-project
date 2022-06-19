; RUN: llc -O1 -mtriple=arm64-apple-ios7.0.0 -enable-andcmp-sinking=true < %s | FileCheck %s
; ModuleID = 'and-cbz-extr-mr.bc'
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64-S128"

define zeroext i1 @foo(i1 %IsEditable, i1 %isTextField, i8* %str1, i8* %str2, i8* %str3, i8* %str4, i8* %str5, i8* %str6, i8* %str7, i8* %str8, i8* %str9, i8* %str10, i8* %str11, i8* %str12, i8* %str13, i32 %int1, i8* %str14) unnamed_addr #0 align 2 {
; CHECK: _foo:
entry:
  %tobool = icmp eq i8* %str14, null
  br i1 %tobool, label %return, label %if.end

; CHECK: %if.end
; CHECK: tbz
if.end:                                           ; preds = %entry
  %and.i.i.i = and i32 %int1, 4
  %tobool.i.i.i = icmp eq i32 %and.i.i.i, 0
  br i1 %tobool.i.i.i, label %if.end12, label %land.rhs.i, !prof !1

land.rhs.i:                                       ; preds = %if.end
  %cmp.i.i.i = icmp eq i8* %str12, %str13
  br i1 %cmp.i.i.i, label %if.then3, label %lor.rhs.i.i.i

lor.rhs.i.i.i:                                    ; preds = %land.rhs.i
  %cmp.i13.i.i.i = icmp eq i8* %str10, %str11
  br i1 %cmp.i13.i.i.i, label %_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit, label %if.end5

_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit: ; preds = %lor.rhs.i.i.i
  %cmp.i.i.i.i = icmp eq i8* %str8, %str9
  br i1 %cmp.i.i.i.i, label %if.then3, label %if.end5

if.then3:                                         ; preds = %_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit, %land.rhs.i
  %tmp11 = load i8, i8* %str14, align 8
  %tmp12 = and i8 %tmp11, 2
  %tmp13 = icmp ne i8 %tmp12, 0
  br label %return

if.end5:                                          ; preds = %_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit, %lor.rhs.i.i.i
; CHECK: %if.end5
; CHECK: tbz
  br i1 %IsEditable, label %if.end12, label %land.rhs.i19, !prof !1

land.rhs.i19:                                     ; preds = %if.end5
  %cmp.i.i.i18 = icmp eq i8* %str6, %str7
  br i1 %cmp.i.i.i18, label %if.then7, label %lor.rhs.i.i.i23

lor.rhs.i.i.i23:                                  ; preds = %land.rhs.i19
  %cmp.i13.i.i.i22 = icmp eq i8* %str3, %str4
  br i1 %cmp.i13.i.i.i22, label %_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit28, label %if.end12

_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit28: ; preds = %lor.rhs.i.i.i23
  %cmp.i.i.i.i26 = icmp eq i8* %str1, %str2
  br i1 %cmp.i.i.i.i26, label %if.then7, label %if.end12

if.then7:                                         ; preds = %_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit28, %land.rhs.i19
  br i1 %isTextField, label %if.then9, label %if.end12

if.then9:                                         ; preds = %if.then7
  %tmp23 = load i8, i8* %str5, align 8
  %tmp24 = and i8 %tmp23, 2
  %tmp25 = icmp ne i8 %tmp24, 0
  br label %return

if.end12:                                         ; preds = %if.then7, %_ZNK7WebCore4Node10hasTagNameERKNS_13QualifiedNameE.exit28, %lor.rhs.i.i.i23, %if.end5, %if.end
  %lnot = xor i1 %IsEditable, true
  br label %return

return:                                           ; preds = %if.end12, %if.then9, %if.then3, %entry
  %retval.0 = phi i1 [ %tmp13, %if.then3 ], [ %tmp25, %if.then9 ], [ %lnot, %if.end12 ], [ true, %entry ]
  ret i1 %retval.0
}

attributes #0 = { nounwind ssp }
!1 = !{!"branch_weights", i32 3, i32 5}
