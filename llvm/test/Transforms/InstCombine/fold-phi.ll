; RUN: opt < %s -instcombine -S | FileCheck %s

; CHECK: no_crash
define float @no_crash(float %a) nounwind {
entry:
  br label %for.body

for.body:
  %sum.057 = phi float [ 0.000000e+00, %entry ], [ %add5, %bb0 ]
  %add5 = fadd float %sum.057, %a    ; PR14592
  br i1 undef, label %bb0, label %end

bb0:
  br label %for.body

end:
  ret float %add5
}

; CHECK-LABEL: @pr21377(
define void @pr21377(i32) {
entry:
  br label %while.body

while.body:                                       ; preds = %if.end, %entry
  %phi1 = phi i64 [ undef, %entry ], [ %or2, %if.end ]
  %zext = zext i32 %0 to i64
  br i1 undef, label %if.end, label %if.else

if.else:                                          ; preds = %while.body
  %or1 = or i64 %phi1, %zext
  %and = and i64 %or1, 4294967295
  br label %if.end

if.end:                                           ; preds = %if.else, %while.body
  %phi2 = phi i64 [ %and, %if.else ], [ undef, %while.body ]
  %or2 = or i64 %phi2, %zext
  br label %while.body
}
