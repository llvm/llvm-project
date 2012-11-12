; RUN: llc < %s -mcpu=generic -mtriple=x86_64-apple-macosx | FileCheck %s
; PR10221

;; The registers %x and %y must both spill across the finit call.
;; Check that they are spilled early enough that not copies are needed for the
;; fadd and fpext.

; CHECK: pr10221
; CHECK-NOT: movaps
; CHECK:      movss
; CHECK-NEXT: movss
; CHECK-NEXT: addss
; CHECK-NEXT: cvtss2sd
; CHECK-NEXT: finit

define i32 @pr10221(float %x, float %y, i8** nocapture %_retval) nounwind uwtable ssp {
entry:
  %add = fadd float %x, %y
  %conv = fpext float %add to double
  %call = tail call i32 @finit(double %conv) nounwind
  %tobool = icmp eq i32 %call, 0
  br i1 %tobool, label %return, label %if.end

if.end:                                           ; preds = %entry
  tail call void @foo(float %x, float %y) nounwind
  br label %return

return:                                           ; preds = %entry, %if.end
  %retval.0 = phi i32 [ 0, %if.end ], [ 5, %entry ]
  ret i32 %retval.0
}

declare i32 @finit(double)

declare void @foo(float, float)
