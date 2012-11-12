; RUN: llc < %s -march=ppc64 | FileCheck %s

define i64 @exchange_and_add(i64* %mem, i64 %val) nounwind {
; CHECK: exchange_and_add:
; CHECK: ldarx
  %tmp = atomicrmw add i64* %mem, i64 %val monotonic
; CHECK: stdcx.
  ret i64 %tmp
}

define i64 @exchange_and_cmp(i64* %mem) nounwind {
; CHECK: exchange_and_cmp:
; CHECK: ldarx
  %tmp = cmpxchg i64* %mem, i64 0, i64 1 monotonic
; CHECK: stdcx.
; CHECK: stdcx.
  ret i64 %tmp
}

define i64 @exchange(i64* %mem, i64 %val) nounwind {
; CHECK: exchange:
; CHECK: ldarx
  %tmp = atomicrmw xchg i64* %mem, i64 1 monotonic
; CHECK: stdcx.
  ret i64 %tmp
}
