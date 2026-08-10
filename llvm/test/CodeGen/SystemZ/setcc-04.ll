; Test SETCC with an i64 result for every floating-point condition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 | FileCheck %s

; Test CC in { 0 }
define i64 @f1(float %a, float %b) {
; CHECK-LABEL: f1:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], -268435456
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp oeq float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 1 }
define i64 @f2(float %a, float %b) {
; CHECK-LABEL: f2:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 268435456
; CHECK-NEXT: afi [[REG]], -268435456
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp olt float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 1 }
define i64 @f3(float %a, float %b) {
; CHECK-LABEL: f3:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], -536870912
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp ole float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 2 }
define i64 @f4(float %a, float %b) {
; CHECK-LABEL: f4:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 268435456
; CHECK-NEXT: afi [[REG]], 1342177280
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp ogt float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 2 }
define i64 @f5(float %a, float %b) {
; CHECK-LABEL: f5:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 4294967295
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 36
; CHECK: br %r14
  %cond = fcmp oge float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 1, 2 }
define i64 @f6(float %a, float %b) {
; CHECK-LABEL: f6:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], 268435456
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %cond = fcmp one float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 1, 2 }
define i64 @f7(float %a, float %b) {
; CHECK-LABEL: f7:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], -805306368
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp ord float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 3 }
define i64 @f8(float %a, float %b) {
; CHECK-LABEL: f8:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], 1342177280
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp uno float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 3 }
define i64 @f9(float %a, float %b) {
; CHECK-LABEL: f9:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], -268435456
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %cond = fcmp ueq float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 1, 3 }
define i64 @f10(float %a, float %b) {
; CHECK-LABEL: f10:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 36
; CHECK: br %r14
  %cond = fcmp ult float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 1, 3 }
define i64 @f11(float %a, float %b) {
; CHECK-LABEL: f11:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 268435456
; CHECK-NEXT: afi [[REG]], -805306368
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp ule float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 2, 3 }
define i64 @f12(float %a, float %b) {
; CHECK-LABEL: f12:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 35
; CHECK: br %r14
  %cond = fcmp ugt float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 0, 2, 3 }
define i64 @f13(float %a, float %b) {
; CHECK-LABEL: f13:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: xilf [[REG]], 268435456
; CHECK-NEXT: afi [[REG]], 1879048192
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp uge float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}

; Test CC in { 1, 2, 3 }
define i64 @f14(float %a, float %b) {
; CHECK-LABEL: f14:
; CHECK: ipm [[REG:%r[0-5]]]
; CHECK-NEXT: afi [[REG]], 1879048192
; CHECK-NEXT: risbg %r2, [[REG]], 63, 191, 33
; CHECK: br %r14
  %cond = fcmp une float %a, %b
  %res = zext i1 %cond to i64
  ret i64 %res
}
