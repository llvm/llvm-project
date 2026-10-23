; RUN: opt < %s -S -passes=instsimplify -max-assumes-per-value=3 | FileCheck %s --check-prefixes=CHECK,USED
; RUN: opt < %s -S -passes=instsimplify -max-assumes-per-value=2 | FileCheck %s --check-prefixes=CHECK,IGNORED

; Analyses inspect every assumption cached for a value, so only the first
; -max-assumes-per-value assumptions affecting it are cached. Here the
; assumption that proves the comparison is the third one.

declare void @llvm.assume(i1)

define i1 @assumes_within_limit(i32 %x) {
; CHECK-LABEL: define i1 @assumes_within_limit(
; USED: ret i1 true
; IGNORED: ret i1 %cmp
  %u1 = icmp ne i32 %x, 1234
  call void @llvm.assume(i1 %u1)
  %u2 = icmp ne i32 %x, 5678
  call void @llvm.assume(i1 %u2)
  %c = icmp sgt i32 %x, 41
  call void @llvm.assume(i1 %c)
  %cmp = icmp sgt i32 %x, 0
  ret i1 %cmp
}

; The limit applies per value, so assumptions about %y are still cached when %x
; is affected by more of them than the limit allows.
define i1 @limit_is_per_value(i32 %x, i32 %y) {
; CHECK-LABEL: define i1 @limit_is_per_value(
; CHECK: ret i1 true
  %cx = icmp sgt i32 %x, 41
  call void @llvm.assume(i1 %cx)
  call void @llvm.assume(i1 %cx)
  call void @llvm.assume(i1 %cx)
  %cy = icmp sgt i32 %y, 41
  call void @llvm.assume(i1 %cy)
  %cmp = icmp sgt i32 %y, 0
  ret i1 %cmp
}
