; RUN: llc -mtriple=nanomips -verify-machineinstrs -enable-machine-outliner < %s | FileCheck %s

define dso_local i32 @testing_tailcall(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
  %cmp = icmp slt i32 %x, %y
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %mul = mul nsw i32 %a, 10
  %add = add nsw i32 %mul, %y
  %add3 = add nsw i32 %add, 11
  br label %if.end37

if.else:                                          ; preds = %entry
  %cmp4 = icmp sgt i32 %x, %a
  br i1 %cmp4, label %if.then5, label %if.else11

if.then5:                                         ; preds = %if.else
; CHECK: bc OUTLINED_FUNCTION_1
  %mul6 = mul nsw i32 %a, 10
  %add7 = add nsw i32 %mul6, %y
  %reass.add8587 = add i32 %add7, %a
  %add10 = shl i32 %reass.add8587, 1
  %sub = add nsw i32 %add10, -10
  br label %if.end37

if.else11:                                        ; preds = %if.else
  %cmp12 = icmp slt i32 %y, %x
  br i1 %cmp12, label %if.then13, label %if.else20

if.then13:                                        ; preds = %if.else11
  %mul14 = mul nsw i32 %a, 10
  %add15 = add nsw i32 %mul14, %y
  %reass.add8284 = add i32 %add15, %a
  %add18 = shl i32 %reass.add8284, 1
  %sub19 = add nsw i32 %add18, -22
  br label %if.end37

if.else20:                                        ; preds = %if.else11
  %cmp21 = icmp eq i32 %x, %a
  br i1 %cmp21, label %if.then22, label %if.else29

if.then22:                                        ; preds = %if.else20
  %mul23 = mul nsw i32 %x, 10
  %add24 = add nsw i32 %mul23, %y
  %reass.add7981 = add i32 %add24, %x
  %add27 = shl i32 %reass.add7981, 1
  %sub28 = add nsw i32 %add27, -20
  br label %if.end37

if.else29:                                        ; preds = %if.else20
  %mul30 = mul nsw i32 %a, 10
  %add31 = add nsw i32 %mul30, %y
  %reass.add = shl i32 %a, 1
  %add33 = add i32 %add31, %reass.add
  %mul34 = shl nsw i32 %add33, 1
  br label %if.end37

if.end37:                                         ; preds = %if.then5, %if.then22, %if.else29, %if.then13, %if.then
  %x.addr.0 = phi i32 [ %add, %if.then ], [ %add7, %if.then5 ], [ %add15, %if.then13 ], [ %add24, %if.then22 ], [ %add31, %if.else29 ]
  %a.addr.0 = phi i32 [ %add3, %if.then ], [ %sub, %if.then5 ], [ %sub19, %if.then13 ], [ %sub28, %if.then22 ], [ %mul34, %if.else29 ]
  %add38 = add nsw i32 %a.addr.0, %x.addr.0
  ret i32 %add38
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local i32 @testing_default1(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
; CHECK: save 16, $ra
; CHECK: balc OUTLINED_FUNCTION_0
; CHECK: restore 16, $ra
  %mul = mul nsw i32 %a, 50
  %add = add nsw i32 %mul, %y
  %sub = sub nsw i32 %a, %add
  %add1 = add nsw i32 %sub, 50
  %mul2 = mul nsw i32 %add1, %add
  %add4 = add nsw i32 %mul2, %add
  ret i32 %add4
}

; Function Attrs: nofree norecurse nosync nounwind readnone
define dso_local void @doing_something(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr #1 {
entry:
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define dso_local i32 @testing_default2(i32 signext %x, i32 signext %y, i32 signext %a) local_unnamed_addr #0 {
entry:
; CHECK: save 16, $ra
; CHECK: balc OUTLINED_FUNCTION_0
; CHECK: restore 16, $ra
  %mul = mul nsw i32 %a, 50
  %add = add nsw i32 %mul, %y
  %sub = sub nsw i32 %a, %add
  %add1 = add nsw i32 %sub, 50
  %mul2 = mul nsw i32 %add1, %add
  %sub4 = sub nsw i32 %mul2, %add
  ret i32 %sub4
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind readnone willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i7200" "target-features"="+i7200,+soft-float,-noabicalls" "use-soft-float"="true" }
attributes #1 = { nofree norecurse nosync nounwind readnone "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="i7200" "target-features"="+i7200,+soft-float,-noabicalls" "use-soft-float"="true" }

