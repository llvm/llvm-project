; RUN: llc < %s -mtriple=armv7-apple-darwin -arm-atomic-cfg-tidy=0 | FileCheck %s

define double @f1() nounwind {
; CHECK-LABEL: f1:
; CHECK: .data_region
; CHECK: .long 1413754129
; CHECK: .long 1074340347
; CHECK: .end_data_region
  ret double 0x400921FB54442D11
}


define i32 @f2()  {
; CHECK-LABEL: f2:
; CHECK: .data_region jt32
; CHECK: .end_data_region

entry:
  switch i32 undef, label %return [
    i32 1, label %sw.bb
    i32 2, label %sw.bb6
    i32 3, label %sw.bb13
    i32 4, label %sw.bb20
  ]

sw.bb:                                            ; preds = %entry
  br label %return

sw.bb6:                                           ; preds = %entry
  br label %return

sw.bb13:                                          ; preds = %entry
  br label %return

sw.bb20:                                          ; preds = %entry
  %div = sdiv i32 undef, undef
  br label %return

return:                                           ; preds = %sw.bb20, %sw.bb13, %sw.bb6, %sw.bb, %entry
  %retval.0 = phi i32 [ %div, %sw.bb20 ], [ undef, %sw.bb13 ], [ undef, %sw.bb6 ], [ undef, %sw.bb ], [ 0, %entry ]
  ret i32 %retval.0
}
