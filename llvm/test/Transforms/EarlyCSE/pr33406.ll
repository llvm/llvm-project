; RUN: opt -early-cse-memssa -earlycse-debug-hash -S %s | FileCheck %s

; CHECK: define void @patatino() {
; CHECK:  for.cond:
; CHECK-NEXT:  br i1 true, label %if.end, label %for.inc
; CHECK:  if.end:
; CHECK-NEXT:  %tinkywinky = load i32, i32* @b
; CHECK-NEXT:  br i1 true, label %for.inc, label %for.inc
; CHECK:  for.inc:
; CHECK-NEXT:  ret void


@b = external global i32

define void @patatino() {
for.cond:
  br i1 true, label %if.end, label %for.inc

if.end:
  %tinkywinky = load i32, i32* @b
  store i32 %tinkywinky, i32* @b
  br i1 true, label %for.inc, label %for.inc

for.inc:
  ret void
}
