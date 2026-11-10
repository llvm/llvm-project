; RUN: opt -passes=correlated-propagation -pass-remarks-output=%t.opt.yaml -disable-output %s
; RUN: FileCheck %s --input-file=%t.opt.yaml --implicit-check-not="--- !"

; LVI only looks at a limited number of uses for conditions that constrain a value at a use. Reaching that limit emits a missed remark.

; The select condition is too many uses away from the smin to be considered.

; CHECK:      --- !Missed
; CHECK-NEXT: Pass:            lazy-value-info
; CHECK-NEXT: Name:            UseLimitReached
; CHECK-NEXT: Function:        over_limit
; CHECK-NEXT: Args:
; CHECK-NEXT:   - String:          'stopped looking for conditions on '
; CHECK-NEXT:   - Value:           m
; CHECK-NEXT:   - String:          ' after '
; CHECK-NEXT:   - MaxUses:         '3'
; CHECK-NEXT:   - String:          ' transitive uses'
; CHECK-NEXT: ...
define i32 @over_limit(i32 %m) {
  %cmp = icmp slt i32 %m, 0
  %min = call i32 @llvm.smin.i32(i32 %m, i32 0)
  %a = xor i32 %min, 1
  %b = xor i32 %a, 2
  %c = xor i32 %b, 3
  %sel = select i1 %cmp, i32 %c, i32 0
  ret i32 %sel
}

; The select is reached within the limit, no remark.
define i32 @within_limit(i32 %m) {
  %cmp = icmp slt i32 %m, 0
  %min = call i32 @llvm.smin.i32(i32 %m, i32 0)
  %a = xor i32 %min, 1
  %sel = select i1 %cmp, i32 %a, i32 0
  ret i32 %sel
}
