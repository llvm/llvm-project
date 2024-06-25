; RUN: llc %s -x86-early-ifcvt -pass-remarks='early-ifcvt' -pass-remarks-missed='early-ifcvt' -mcpu=k8 -o - 2>&1 | FileCheck %s
target triple = "x86_64-none-none"

; CHECK: remark: <unknown>:0:0: performing if-conversion on branch:
; CHECK-SAME: the condition adds {{[0-9]+}} cycle{{s?}} to the critical path,
; CHECK-SAME: and the short leg adds another {{[0-9]+}} cycles{{s?}},
; CHECK-SAME: and the long leg adds another {{[0-9]+}} cycles{{s?}},
; CHECK-SAME: each staying under the threshold of {{[0-9]+}} cycles{{s?}}.
define i32 @mm1(i1 %pred, i32 %val) {
entry:
  br i1 %pred, label %if.true, label %if.else

if.true:
  %v1 = add i32 1, %val
  br label %if.else

if.else:
  %res = phi i32 [ %val, %entry ], [ %v1, %if.true ]
  ret i32 %res
}

; CHECK: remark: <unknown>:0:0: did not if-convert branch:
; CHECK-SAME: the condition would add {{[0-9]+}} cycles{{s?}} to the critical path,
; CHECK-SAME: and the short leg would add another {{[0-9]+}} cycles{{s?}},
; CHECK-SAME: and the long leg would add another {{[0-9]+}} cycles{{s?}} exceeding the limit of {{[0-9]+}} cycles{{s?}}.
define i32 @mm2(i1 %pred, i32 %val, i32 %e1, i32 %e2, i32 %e3, i32 %e4, i32 %e5) {
entry:
  br i1 %pred, label %if.true, label %if.else

if.true:
  %v1 = add i32 %e1, %val
  %v2 = add i32 %e2, %v1
  %v3 = add i32 %e3, %v2
  %v4 = add i32 %e4, %v3
  br label %if.else

if.else:
  %res = phi i32 [ %val, %entry ], [ %v4, %if.true ]
  ret i32 %res
}

; CHECK: did not if-convert branch:
; CHECK-SAME: the resulting critical path ({{[0-9]+}} cycles{{s?}})
; CHECK-SAME: would extend the shorter leg's critical path ({{[0-9]+}} cycle{{s?}})
; CHECK-SAME: by more than the threshold of {{[0-9]+}} cycles{{s?}},
; CHECK-SAME: which cannot be hidden by available ILP.
define i32 @mm3(i1 %pred, i32 %val, i32 %e1, i128 %e2, i128 %e3, i128 %e4, i128 %e5) {
entry:
  br i1 %pred, label %if.true, label %if.false

if.true:
  br label %if.endif

if.false:
  %f1 = mul i32 %e1, %e1
  %f3 = sext i32 %f1 to i128
  %f4 = mul i128 %e2, %f3
  %f6 = add i128 %e3, %f4
  %f7 = xor i128 %e4, %f6
  %f8 = xor i128 %e5, %f7
  %a1 = ashr i128 %f8, %e5
  %f5 = trunc i128 %a1 to i32
  br label %if.endif

if.endif:
  %r1 = phi i32 [ %val, %if.true ], [ %f1, %if.false ]
  %r2 = phi i32 [ %val, %if.true ], [ %f5, %if.false ]
  %res = add i32 %r1, %r2
  ret i32 %res
}
