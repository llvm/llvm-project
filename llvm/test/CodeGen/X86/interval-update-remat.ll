; RUN: llc -verify-regalloc -verify-machineinstrs < %s
; PR27275: When enabling remat for vreg defined by PHIs, make sure the update
; of the live range removes dead phi. Otherwise, we may end up with PHIs with
; incorrect operands and that will trigger assertions or verifier failures
; in later passes.

target datalayout = "e-m:e-p:32:32-f64:32:64-f80:32-n8:16:32-S128"
target triple = "i386-unknown-linux-gnu"

@b = external global i64, align 8
@d = external global i32, align 4
@e = external global i64, align 8
@h = external global i16, align 2
@a = external global i8, align 1
@g = external global i64, align 8
@j = external global i32, align 4
@f = external global i16, align 2
@.str = external unnamed_addr constant [12 x i8], align 1

define void @fn1(i1 %arg) {
entry:
  %tmp = load i64, ptr @b, align 8
  %or = or i64 0, 3299921317
  %and = and i64 %or, %tmp
  %tmp1 = load i32, ptr @d, align 4
  br i1 %arg, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %entry
  %tobool3 = icmp ne i8 undef, 0
  br label %lor.end

lor.end:                                          ; preds = %lor.rhs, %entry
  %lor.ext = zext i1 undef to i32
  %tmp2 = load i64, ptr @e, align 8
  br i1 %arg, label %lor.rhs5, label %lor.end7

lor.rhs5:                                         ; preds = %lor.end
  br label %lor.end7

lor.end7:                                         ; preds = %lor.rhs5, %lor.end
  %tmp3 = phi i1 [ true, %lor.end ], [ false, %lor.rhs5 ]
  %neg13 = xor i64 %tmp, -1
  %conv25 = zext i1 %tmp3 to i32
  %tobool46 = icmp eq i64 %tmp, 0
  %.pre = load i16, ptr @h, align 2
  %tobool10 = icmp eq i16 %.pre, 0
  %neg.us = xor i32 %tmp1, -1
  %conv12.us = sext i32 %neg.us to i64
  %tobool23.us = icmp eq i64 %tmp2, %and
  %conv39.us = sext i32 %tmp1 to i64
  br label %LABEL_mSmSDb

LABEL_mSmSDb.loopexit:                            ; preds = %lor.end32.us
  %conv42.us.lcssa = phi i32 [ %conv42.us, %lor.end32.us ]
  store i64 undef, ptr @g, align 8
  br label %LABEL_mSmSDb

LABEL_mSmSDb:                                     ; preds = %LABEL_mSmSDb.loopexit, %lor.end7
  %tmp4 = phi i32 [ undef, %lor.end7 ], [ %conv42.us.lcssa, %LABEL_mSmSDb.loopexit ]
  %tmp5 = phi i64 [ %tmp, %lor.end7 ], [ 0, %LABEL_mSmSDb.loopexit ]
  br i1 %tobool10, label %LABEL_BRBRN.preheader, label %if.then

if.then:                                          ; preds = %LABEL_mSmSDb
  store i8 undef, ptr @a, align 1
  br label %LABEL_BRBRN.preheader

LABEL_BRBRN.preheader:                            ; preds = %if.then, %LABEL_mSmSDb
  %.pre63 = load i64, ptr @g, align 8
  br i1 %tobool46, label %LABEL_BRBRN.us, label %LABEL_BRBRN.outer

LABEL_BRBRN.outer:                                ; preds = %if.then47, %LABEL_BRBRN.preheader
  %.ph = phi i32 [ 0, %if.then47 ], [ %tmp4, %LABEL_BRBRN.preheader ]
  %.ph64 = phi i32 [ %conv50, %if.then47 ], [ %tmp1, %LABEL_BRBRN.preheader ]
  %.ph65 = phi i64 [ %tmp16, %if.then47 ], [ %.pre63, %LABEL_BRBRN.preheader ]
  %.ph66 = phi i64 [ 0, %if.then47 ], [ %tmp2, %LABEL_BRBRN.preheader ]
  %.ph67 = phi i64 [ %.pre56.pre, %if.then47 ], [ %tmp5, %LABEL_BRBRN.preheader ]
  %neg = xor i32 %.ph64, -1
  %conv12 = sext i32 %neg to i64
  %tobool23 = icmp eq i64 %.ph66, %and
  %tmp6 = load i32, ptr @j, align 4
  %shr = lshr i32 %conv25, %tmp6
  %conv39 = sext i32 %.ph64 to i64
  br label %LABEL_BRBRN

LABEL_BRBRN.us:                                   ; preds = %lor.end32.us, %LABEL_BRBRN.preheader
  %tmp7 = phi i32 [ %conv42.us, %lor.end32.us ], [ %tmp4, %LABEL_BRBRN.preheader ]
  %tmp8 = phi i64 [ undef, %lor.end32.us ], [ %.pre63, %LABEL_BRBRN.preheader ]
  %tmp9 = phi i64 [ %tmp10, %lor.end32.us ], [ %tmp5, %LABEL_BRBRN.preheader ]
  %mul.us = mul i64 %tmp8, %neg13
  %mul14.us = mul i64 %mul.us, %conv12.us
  %cmp.us = icmp sgt i64 %tmp2, %mul14.us
  %conv16.us = zext i1 %cmp.us to i64
  %xor.us = xor i64 %conv16.us, %tmp9
  %rem18.us = urem i32 %lor.ext, %tmp7
  %conv19.us = zext i32 %rem18.us to i64
  br i1 %tobool23.us, label %lor.rhs24.us, label %lor.end32.us

lor.rhs24.us:                                     ; preds = %LABEL_BRBRN.us
  br label %lor.end32.us

lor.end32.us:                                     ; preds = %lor.rhs24.us, %LABEL_BRBRN.us
  %tmp10 = phi i64 [ -2, %LABEL_BRBRN.us ], [ -1, %lor.rhs24.us ]
  %xor.us.not = xor i64 %xor.us, -1
  %neg36.us = and i64 %conv19.us, %xor.us.not
  %conv37.us = zext i32 %tmp7 to i64
  %sub38.us = sub nsw i64 %neg36.us, %conv37.us
  %mul40.us = mul nsw i64 %sub38.us, %conv39.us
  %neg41.us = xor i64 %mul40.us, 4294967295
  %conv42.us = trunc i64 %neg41.us to i32
  %tobool43.us = icmp eq i8 undef, 0
  br i1 %tobool43.us, label %LABEL_mSmSDb.loopexit, label %LABEL_BRBRN.us

LABEL_BRBRN:                                      ; preds = %lor.end32, %LABEL_BRBRN.outer
  %tmp11 = phi i32 [ %conv42, %lor.end32 ], [ %.ph, %LABEL_BRBRN.outer ]
  %tmp12 = phi i64 [ %neg21, %lor.end32 ], [ %.ph65, %LABEL_BRBRN.outer ]
  %tmp13 = phi i64 [ %conv35, %lor.end32 ], [ %.ph67, %LABEL_BRBRN.outer ]
  %mul = mul i64 %tmp12, %neg13
  %mul14 = mul i64 %mul, %conv12
  %cmp = icmp sgt i64 %.ph66, %mul14
  %conv16 = zext i1 %cmp to i64
  %xor = xor i64 %conv16, %tmp13
  %rem18 = urem i32 %lor.ext, %tmp11
  %conv19 = zext i32 %rem18 to i64
  %neg21 = or i64 %xor, undef
  br i1 %tobool23, label %lor.rhs24, label %lor.end32

lor.rhs24:                                        ; preds = %LABEL_BRBRN
  %tmp14 = load volatile i16, ptr @f, align 2
  %conv26 = sext i16 %tmp14 to i32
  %and27 = and i32 %conv26, %shr
  %conv28 = sext i32 %and27 to i64
  %mul29 = mul nsw i64 %conv28, %tmp
  %and30 = and i64 %mul29, %tmp13
  %tobool31 = icmp ne i64 %and30, 0
  br label %lor.end32

lor.end32:                                        ; preds = %lor.rhs24, %LABEL_BRBRN
  %tmp15 = phi i1 [ true, %LABEL_BRBRN ], [ %tobool31, %lor.rhs24 ]
  %lor.ext33 = zext i1 %tmp15 to i32
  %neg34 = xor i32 %lor.ext33, -1
  %conv35 = sext i32 %neg34 to i64
  %xor.not = xor i64 %xor, -1
  %neg36 = and i64 %conv19, %xor.not
  %conv37 = zext i32 %tmp11 to i64
  %sub38 = sub nsw i64 %neg36, %conv37
  %mul40 = mul nsw i64 %sub38, %conv39
  %neg41 = xor i64 %mul40, 4294967295
  %conv42 = trunc i64 %neg41 to i32
  %tobool43 = icmp eq i8 undef, 0
  br i1 %tobool43, label %if.then47, label %LABEL_BRBRN

if.then47:                                        ; preds = %lor.end32
  tail call void (ptr, ...) @printf(ptr @.str, i64 %conv39)
  %tmp16 = load i64, ptr @g, align 8
  %neg49 = xor i64 %tmp16, 4294967295
  %conv50 = trunc i64 %neg49 to i32
  %.pre56.pre = load i64, ptr @b, align 8
  br label %LABEL_BRBRN.outer
}

declare void @printf(ptr nocapture readonly, ...)
