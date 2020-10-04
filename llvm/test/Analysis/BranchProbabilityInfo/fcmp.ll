; RUN: opt < %s -analyze -branch-prob -enable-new-pm=0 | FileCheck %s
; RUN: opt < %s -passes='print<branch-prob>' -disable-output 2>&1 | FileCheck %s

; This function tests the floating point unorder comparison. The probability
; of NaN should be extremely small.
; CHECK: Printing analysis {{.*}} for function 'uno'
; CHECK:  edge  -> a probability is 0x00000800 / 0x80000000 = 0.00%
; CHECK:  edge  -> b probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]

define void @uno(float %val1, float %val2) {
  %cond = fcmp uno float %val1, %val2
  br i1 %cond, label %a, label %b

a:
  call void @fa()
  ret void

b:
  call void @fb()
  ret void
}

; This function tests the floating point order comparison.
; CHECK: Printing analysis {{.*}} for function 'ord'
; CHECK:  edge  -> a probability is 0x7ffff800 / 0x80000000 = 100.00% [HOT edge]
; CHECK:  edge  -> b probability is 0x00000800 / 0x80000000 = 0.00%

define void @ord(float %val1, float %val2) {
  %cond = fcmp ord float %val1, %val2
  br i1 %cond, label %a, label %b

a:
  call void @fa()
  ret void

b:
  call void @fb()
  ret void
}

declare void @fa()
declare void @fb()
