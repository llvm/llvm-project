; RUN: opt < %s -instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-n8:16:32:64"

define i64 @sink_i1_casts(i1 %cond1, i1 %cond2) {
; CHECK-LABEL: @sink_i1_casts(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[Z1:%.*]] = zext i1 %cond1 to i64
; CHECK-NEXT:    br i1 %cond1, label %if, label %end
; CHECK:       if:
; CHECK-NEXT:    [[Z2:%.*]] = zext i1 %cond2 to i64
; CHECK-NEXT:    br label %end
; CHECK:       end:
; CHECK-NEXT:    [[PHI:%.*]] = phi i64 [ [[Z1]], %entry ], [ [[Z2]], %if ]
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

