; Test an i32 0/-1 SELECTCCC for every floating-point condition.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

; Test CC in { 1, 2, 3 }
define i32 @f1(float %a, float %b) {
; CHECK-LABEL: f1:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, 1879048192
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp oeq float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0, 2, 3 }
define i32 @f2(float %a, float %b) {
; CHECK-LABEL: f2:
; CHECK: ipm %r2
; CHECK-NEXT: xilf %r2, 268435456
; CHECK-NEXT: afi %r2, 1879048192
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp olt float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 2, 3 }
define i32 @f3(float %a, float %b) {
; CHECK-LABEL: f3:
; CHECK: ipm %r2
; CHECK-NEXT: sll %r2, 2
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ole float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0, 1, 3 }
define i32 @f4(float %a, float %b) {
; CHECK-LABEL: f4:
; CHECK: ipm %r2
; CHECK-NEXT: xilf %r2, 268435456
; CHECK-NEXT: afi %r2, -805306368
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ogt float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 1, 3 }
define i32 @f5(float %a, float %b) {
; CHECK-LABEL: f5:
; CHECK: ipm %r2
; CHECK-NEXT: sll %r2, 3
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp oge float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0, 3 }
define i32 @f6(float %a, float %b) {
; CHECK-LABEL: f6:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: sll %r2, 2
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp one float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 3 }
define i32 @f7(float %a, float %b) {
; CHECK-LABEL: f7:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, 1342177280
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ord float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0, 1, 2 }
define i32 @f8(float %a, float %b) {
; CHECK-LABEL: f8:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, -805306368
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp uno float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 1, 2 }
define i32 @f9(float %a, float %b) {
; CHECK-LABEL: f9:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, 268435456
; CHECK-NEXT: sll %r2, 2
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ueq float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0, 2 }
define i32 @f10(float %a, float %b) {
; CHECK-LABEL: f10:
; CHECK: ipm %r2
; CHECK-NEXT: xilf %r2, 4294967295
; CHECK-NEXT: sll %r2, 3
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ult float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 2 }
define i32 @f11(float %a, float %b) {
; CHECK-LABEL: f11:
; CHECK: ipm %r2
; CHECK-NEXT: xilf %r2, 268435456
; CHECK-NEXT: afi %r2, 1342177280
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ule float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0, 1 }
define i32 @f12(float %a, float %b) {
; CHECK-LABEL: f12:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, -536870912
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp ugt float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 1 }
define i32 @f13(float %a, float %b) {
; CHECK-LABEL: f13:
; CHECK: ipm %r2
; CHECK-NEXT: xilf %r2, 268435456
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp uge float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}

; Test CC in { 0 }
define i32 @f14(float %a, float %b) {
; CHECK-LABEL: f14:
; CHECK: ipm %r2
; CHECK-NEXT: afi %r2, -268435456
; CHECK-NEXT: sra %r2, 31
; CHECK: br %r14
  %cond = fcmp une float %a, %b
  %res = select i1 %cond, i32 0, i32 -1
  ret i32 %res
}
