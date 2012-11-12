; RUN: opt -inline -instcombine -S < %s
; PR12967

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.7.0"

@d = common global i32 0, align 4
@c = common global i32 0, align 4
@e = common global i32 0, align 4
@f = common global i32 0, align 4
@a = common global i32 0, align 4
@b = common global i32 0, align 4

define signext i8 @fn1(i32 %p1) nounwind uwtable readnone ssp {
entry:
  %shr = lshr i32 1, %p1
  %conv = trunc i32 %shr to i8
  ret i8 %conv
}

define void @fn4() nounwind uwtable ssp {
entry:
  %0 = load i32* @d, align 4, !tbaa !0
  %cmp = icmp eq i32 %0, 0
  %conv = zext i1 %cmp to i32
  store i32 %conv, i32* @c, align 4, !tbaa !0
  tail call void @fn3(i32 %conv) nounwind
  ret void
}

define void @fn3(i32 %p1) nounwind uwtable ssp {
entry:
  %and = and i32 %p1, 8
  store i32 %and, i32* @e, align 4, !tbaa !0
  %sub = add nsw i32 %and, -1
  store i32 %sub, i32* @f, align 4, !tbaa !0
  %0 = load i32* @a, align 4, !tbaa !0
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %1 = load i32* @b, align 4, !tbaa !0
  %.lobit = lshr i32 %1, 31
  %2 = trunc i32 %.lobit to i8
  %.not = xor i8 %2, 1
  br label %if.end

if.else:                                          ; preds = %entry
  %call = tail call signext i8 @fn1(i32 %sub) nounwind
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %storemerge.in = phi i8 [ %call, %if.else ], [ %.not, %if.then ]
  %storemerge = sext i8 %storemerge.in to i32
  store i32 %storemerge, i32* @b, align 4
  ret void
}

!0 = metadata !{metadata !"int", metadata !1}
!1 = metadata !{metadata !"omnipotent char", metadata !2}
!2 = metadata !{metadata !"Simple C/C++ TBAA"}
