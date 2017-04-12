; RUN: opt -licm -S < %s | FileCheck %s

; Function Attrs: noinline norecurse nounwind readnone ssp uwtable
define zeroext i1 @f(double %v) #0 {
entry:
; CHECK-LABEL: @f(
; CHECK-NEXT: entry:
; CHECK-NEXT: fdiv fast double 1.000000e+00, %v
  br label %loop

loop:                                       ; preds = %entry, %loop
  %v3 = phi i32 [ 0, %entry ], [ %v11, %loop ]
  %v4 = phi i32 [ 0, %entry ], [ %v12, %loop ]
  %v5 = uitofp i32 %v4 to double

; CHECK-LABEL: loop:
; CHECK: fmul fast double
; CHECK-NOT: fdiv
  %v6 = fdiv fast double %v5, %v
  %v7 = fptoui double %v6 to i64
  %v8 = and i64 %v7, 1
  %v9 = xor i64 %v8, 1
  %v10 = trunc i64 %v9 to i32
  %v11 = add i32 %v10, %v3
  %v12 = add nuw i32 %v4, 1
  %v13 = icmp eq i32 %v12, -1
  br i1 %v13, label %end, label %loop

end:                                      ; preds = %loop
  %v15 = phi i32 [ %v11, %loop ]
  %v16 = icmp ne i32 %v15, 0
  ret i1 %v16
}

