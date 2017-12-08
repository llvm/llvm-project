; RUN: llc -verify-machineinstrs -mcpu=pwr8 -stop-after codegenprepare -mtriple=powerpc64le-unknown-gnu-linux  < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -stop-after codegenprepare -mtriple=powerpc64-unknown-gnu-linux  < %s | FileCheck %s

; Function Attrs: noinline norecurse nounwind readnone
define hidden signext i32 @call1(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr #0 {
entry:
  %add = add nsw i32 %b, %a
  %add1 = add nsw i32 %add, %c
  ret i32 %add1
}

; Function Attrs: nounwind
define signext i32 @test(i32 signext %a, i32 signext %b, i32 signext %c) local_unnamed_addr #1 {
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %call = tail call signext i32 @call1(i32 signext %a, i32 signext %b, i32 signext %c)
  br label %return
; The return should get duplciated here to enable a tail-call opportunity.
; CHECK-LABEL: if.then:
; CHECK-NEXT:  %[[T1:[a-zA-Z0-9]+]] = tail call signext i32 @call1
; CHECK-NEXT:  ret i32 %[[T1]]

if.end:                                           ; preds = %entry
  %cmp1 = icmp slt i32 %a, %b
  br i1 %cmp1, label %if.then2, label %if.end4

if.then2:                                         ; preds = %if.end
  %call3 = tail call signext i32 @call2(i32 signext %a, i32 signext %b, i32 signext %c) #3
  br label %return
; No duplication here since we cannot tail-call an external function anyway.
; CHECK-LABEL: if.then2:
; CHECK-NEXT:  tail call signext i32 @call2
; CHECK-NEXT:  br

if.end4:                                          ; preds = %if.end
  %cmp5 = icmp sgt i32 %b, %c
  br i1 %cmp5, label %if.then6, label %return

if.then6:                                         ; preds = %if.end4
  %call7 = tail call fastcc signext i32 @call3(i32 signext %a, i32 signext %b, i32 signext %c)
  br label %return
; No duplication here because the calling convention mismatch means we won't tail-call
; CHECK_LABEL: if.then13:
; CHECK:       tail call fastcc signext i32 @call3
; CHECK-NEXT:  br

return:                                           ; preds = %if.end4, %if.then6, %if.then2, %if.then
  %retval.0 = phi i32 [ %call, %if.then ], [ %call3, %if.then2 ], [ %call7, %if.then6 ], [ %c, %if.end4 ]
  ret i32 %retval.0
}

declare signext i32 @call2(i32 signext, i32 signext, i32 signext) local_unnamed_addr #2

; Function Attrs: noinline norecurse nounwind readnone
define internal fastcc signext i32 @call3(i32 signext %a, i32 signext %b, i32 signext %c) unnamed_addr #0 {
entry:
  %mul = mul nsw i32 %b, %a
  %mul1 = mul nsw i32 %mul, %c
  ret i32 %mul1
}
