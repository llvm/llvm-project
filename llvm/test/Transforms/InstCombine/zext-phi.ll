; RUN: opt < %s -passes=instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-n8:16:32:64"

; Although i1 is not in the datalayout, we should treat it
; as a legal type because it is a fundamental type in IR.
; This means we should shrink the phi (sink the zexts).

define i64 @sink_i1_casts(i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @sink_i1_casts(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 %cond1, label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[PHI_IN:%.*]] = phi i1 [ %cond1, %entry ], [ %cond2, %if ]
; CHECK-NEXT:    [[PHI:%.*]] = zext i1 [[PHI_IN]] to i64
; CHECK-NEXT:    ret i64 [[PHI]]
;
entry:
  %z1 = zext i1 %cond1 to i64
  br i1 %cond1, label %if, label %end

if:
  %z2 = zext i1 %cond2 to i64
  br label %end

end:
  %phi = phi i64 [ %z1, %entry ], [ %z2, %if ]
  ret i64 %phi
}

