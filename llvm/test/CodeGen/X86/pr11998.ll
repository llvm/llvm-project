; RUN: llc < %s -mcpu=corei7-avx -mtriple=x86_64-- -mattr=+avx

define void @autogen_51367_5000(i8, i1 %arg) {
BB:
  %B = srem i8 55, %0
  %B9 = shl i8 %B, %B
  br label %CF

CF:                                               ; preds = %CF, %BB
  br i1 %arg, label %CF, label %CF403

CF403:                                            ; preds = %CF403, %CF
  %S44 = icmp eq i8 %B9, %0
  br i1 %S44, label %CF403, label %CF405

CF405:                                            ; preds = %CF405, %CF403
  br label %CF405
}
