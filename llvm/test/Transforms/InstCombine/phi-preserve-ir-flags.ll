; RUN: opt < %s -instcombine -S -o - | FileCheck %s

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"

; CHECK-LABEL: define float @func1(
define float @func1(float %a, float %b, float %c, i1 %cond) {
entry:
  br i1 %cond, label %cond.true, label %cond.false

cond.true:
  %sub0 = fsub fast float %a, %b
  br label %cond.end

cond.false:
  %sub1 = fsub fast float %a, %c
  br label %cond.end

; The fast-math flags should always be transfered if possible.
; CHECK-LABEL: cond.end
; CHECK  [[PHI:%[^ ]*]] = phi float [ %b, %cond.true ], [ %c, %cond.false ]
; CHECK  fsub fast float %a, [[PHI]]
cond.end:
  %e = phi float [ %sub0, %cond.true ], [ %sub1, %cond.false ]
  ret float %e
}

; CHECK-LABEL: define float @func2(
define float @func2(float %a, float %b, float %c, i1 %cond) {
entry:
  br i1 %cond, label %cond.true, label %cond.false

cond.true:
  %sub0 = fsub fast float %a, %b
  br label %cond.end

cond.false:
  %sub1 = fsub float %a, %c
  br label %cond.end

; The fast-math flags should always be transfered if possible.
; CHECK-LABEL: cond.end
; CHECK  [[PHI:%[^ ]*]] = phi float [ %b, %cond.true ], [ %c, %cond.false ]
; CHECK  fsub float %a, [[PHI]]
cond.end:
  %e = phi float [ %sub0, %cond.true ], [ %sub1, %cond.false ]
  ret float %e
}

; CHECK-LABEL: define float @func3(
define float @func3(float %a, float %b, float %c, i1 %cond) {
entry:
  br i1 %cond, label %cond.true, label %cond.false

cond.true:
  %sub0 = fsub fast float %a, 2.0
  br label %cond.end

cond.false:
  %sub1 = fsub fast float %b, 2.0
  br label %cond.end

; CHECK-LABEL: cond.end
; CHECK  [[PHI:%[^ ]*]] = phi float [ %a, %cond.true ], [ %b, %cond.false ]
; CHECK  fadd fast float %a, [[PHI]]
cond.end:
  %e = phi float [ %sub0, %cond.true ], [ %sub1, %cond.false ]
  ret float %e
}

; CHECK-LABEL: define float @func4(
define float @func4(float %a, float %b, float %c, i1 %cond) {
entry:
  br i1 %cond, label %cond.true, label %cond.false

cond.true:
  %sub0 = fsub fast float %a, 2.0
  br label %cond.end

cond.false:
  %sub1 = fsub float %b, 2.0
  br label %cond.end

; CHECK-LABEL: cond.end
; CHECK  [[PHI:%[^ ]*]] = phi float [ %a, %cond.true ], [ %b, %cond.false ]
; CHECK  fadd float %a, [[PHI]]
cond.end:
  %e = phi float [ %sub0, %cond.true ], [ %sub1, %cond.false ]
  ret float %e
}
