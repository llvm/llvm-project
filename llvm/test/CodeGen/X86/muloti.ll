; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
%0 = type { i64, i64 }
%1 = type { i128, i1 }

define %0 @x(i64 %a.coerce0, i64 %a.coerce1, i64 %b.coerce0, i64 %b.coerce1) nounwind uwtable ssp {
; CHECK: x
entry:
  %tmp16 = zext i64 %a.coerce0 to i128
  %tmp11 = zext i64 %a.coerce1 to i128
  %tmp12 = shl nuw i128 %tmp11, 64
  %ins14 = or i128 %tmp12, %tmp16
  %tmp6 = zext i64 %b.coerce0 to i128
  %tmp3 = zext i64 %b.coerce1 to i128
  %tmp4 = shl nuw i128 %tmp3, 64
  %ins = or i128 %tmp4, %tmp6
  %0 = tail call %1 @llvm.smul.with.overflow.i128(i128 %ins14, i128 %ins)
; CHECK: callq   ___muloti4
  %1 = extractvalue %1 %0, 0
  %2 = extractvalue %1 %0, 1
  br i1 %2, label %overflow, label %nooverflow

overflow:                                         ; preds = %entry
  tail call void @llvm.trap()
  unreachable

nooverflow:                                       ; preds = %entry
  %tmp20 = trunc i128 %1 to i64
  %tmp21 = insertvalue %0 undef, i64 %tmp20, 0
  %tmp22 = lshr i128 %1, 64
  %tmp23 = trunc i128 %tmp22 to i64
  %tmp24 = insertvalue %0 %tmp21, i64 %tmp23, 1
  ret %0 %tmp24
}

declare %1 @llvm.smul.with.overflow.i128(i128, i128) nounwind readnone

declare void @llvm.trap() nounwind
